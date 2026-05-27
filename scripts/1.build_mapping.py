"""1. 建全量 mapping：GG2 24.09 genus -> GTDB r220 该属各 species 的代表基因组。

用 metadata 的 gtdb_representative=t 取每 species 的官方代表基因组（不只是任意基因组）。
全量、不设 cap（资源记录完整）；同时输出统计：覆盖率 / RefSeq vs GenBank / 质量 / 真实蛋白数
→ 精确的下载量级 + GPU 量级估计；K_max cap 预览（按质量排序 top-K）。

输入：
  - data/gg2/MCFCorpusV2.gg2.h5ad（var['Genus'] 8114, g__X）
  - data/bacformer_prior/refs/{bac120,ar53}_metadata_r220.tsv.gz
输出：
  - data/bacformer_prior/genus_to_genomes.tsv（token_idx, gg2_genus, gtdb_genus, species,
    accession, source(RS/GB), completeness, contamination, protein_count）
  + stdout 统计
"""
import gzip
import re
from collections import defaultdict

import anndata as ad
import numpy as np

ROOT = "/home/cml_lab/caiqy/project/MCFProjet"
H5AD = f"{ROOT}/data/gg2/MCFCorpusV2.gg2.h5ad"
REFS = f"{ROOT}/data/bacformer_prior/refs"
META = [f"{REFS}/bac120_metadata_r220.tsv.gz", f"{REFS}/ar53_metadata_r220.tsv.gz"]
OUT = f"{ROOT}/data/bacformer_prior/genus_to_genomes.tsv"

# 实测每蛋白字节数（8 基因组样本）：原始 ~400 B/蛋白，压缩 ~210 B/蛋白
B_RAW, B_ZIP = 400, 210


def strip_pre(x):
    return x[3:] if x[:3] in ("g__", "s__") else x


def normalize(g):
    return re.sub(r"(_\d+)+$", "", g)


def base_name(g):
    return re.sub(r"(_[A-Z])+$", "", normalize(g))


def main():
    a = ad.read_h5ad(H5AD, backed="r")
    gg2 = a.var["Genus"].astype(str).values
    observed = a.var["observed"].astype(bool).values
    a.file.close()
    V = len(gg2)

    # ---- 读 metadata reps ----
    # 列索引（已确认）：0 accession, 19 gtdb_taxonomy, 18 gtdb_representative,
    #                   2 checkm2_completeness, 3 checkm2_contamination, 91 protein_count
    reps = defaultdict(list)  # 精确 gtdb genus -> [(species, acc, src, comp, cont, nprot)]
    n_reps = 0
    for path in META:
        with gzip.open(path, "rt") as f:
            f.readline()  # header
            for line in f:
                c = line.rstrip("\n").split("\t")
                if c[18] != "t":  # gtdb_representative
                    continue
                n_reps += 1
                acc = c[0]
                parts = c[19].split(";")
                if len(parts) < 7:
                    continue
                g = normalize(strip_pre(parts[5]))
                s = strip_pre(parts[6])
                src = "RS" if acc.startswith("RS_") else ("GB" if acc.startswith("GB_") else "?")
                try:
                    comp = float(c[2]) if c[2] else np.nan
                    cont = float(c[3]) if c[3] else np.nan
                except ValueError:
                    comp = cont = np.nan
                try:
                    nprot = int(c[91]) if c[91] and c[91] != "none" else 0
                except ValueError:
                    nprot = 0
                reps[g].append((s, acc, src, comp, cont, nprot))
    print(f"GTDB r220 species 代表基因组数 = {n_reps:,}; distinct genus = {len(reps):,}")

    # ---- 匹配词表，写全量 mapping ----
    rows = []
    matched = 0
    for i, g_raw in enumerate(gg2):
        gn = normalize(strip_pre(g_raw))
        lst = reps.get(gn) or reps.get(base_name(g_raw)) or []
        if lst:
            matched += 1
        for (s, acc, src, comp, cont, nprot) in lst:
            rows.append((i, g_raw, gn, s, acc, src, comp, cont, nprot))

    with open(OUT, "w") as fo:
        fo.write("token_idx\tgg2_genus\tgtdb_genus\tspecies\taccession\tsource\tcompleteness\tcontamination\tprotein_count\n")
        for r in rows:
            fo.write("\t".join(str(x) for x in r) + "\n")
    print(f"写出 {OUT}: {len(rows):,} 行（{matched}/{V} genus 有代表基因组）")

    # ---- 统计 ----
    src_arr = np.array([r[5] for r in rows])
    nprot_arr = np.array([r[8] for r in rows])
    print(f"\n=== RefSeq vs GenBank（决定有无现成 .faa / 要不要 Prodigal）===")
    rs = (src_arr == "RS").sum(); gb = (src_arr == "GB").sum()
    print(f"  RefSeq(RS, 有现成蛋白注释) = {rs:,} ({rs/len(rows)*100:.1f}%)")
    print(f"  GenBank(GB, 可能无注释→Prodigal) = {gb:,} ({gb/len(rows)*100:.1f}%)")
    # 完全没有 RefSeq 代表的 genus（这些属若要全靠 GB → 需 Prodigal）
    g_has_rs = defaultdict(bool)
    for r in rows:
        if r[5] == "RS":
            g_has_rs[r[0]] = True
    g_all = set(r[0] for r in rows)
    only_gb = [g for g in g_all if not g_has_rs[g]]
    print(f"  完全无 RefSeq 代表的 genus = {len(only_gb)}（仅这些属可能需 Prodigal）")

    print(f"\n=== 蛋白数 / 下载量 / GPU 量级 ===")
    tot = int(nprot_arr.sum())
    print(f"  全量代表基因组 {len(rows):,} 个，总蛋白 {tot:,}")
    print(f"  下载(全量): ~{tot*B_RAW/1e9:.0f} GB 原始 / ~{tot*B_ZIP/1e9:.0f} GB 压缩")
    for tp in (1000, 2000):
        print(f"  GPU(全量, @{tp} 蛋白/s): ~{tot/tp/3600:.0f} GPU 时")

    # ---- K_max cap 预览（每属按质量 completeness-5*contamination 排序取 top-K）----
    by_g = defaultdict(list)
    for r in rows:
        q = (r[6] if not np.isnan(r[6]) else 0) - 5 * (r[7] if not np.isnan(r[7]) else 0)
        by_g[r[0]].append((q, r[8]))
    print(f"\n=== K_max cap 预览（每属按质量 top-K）===")
    print(f"  {'K_max':>6s} {'基因组数':>10s} {'总蛋白':>14s} {'下载压缩':>10s} {'GPU@1500/s':>12s}")
    for k in (None, 64, 32, 16):
        ng = tp_ = 0
        for g, lst in by_g.items():
            sel = sorted(lst, reverse=True)[:k] if k else lst
            ng += len(sel)
            tp_ += sum(n for _, n in sel)
        label = "全量" if k is None else str(k)
        print(f"  {label:>6s} {ng:>10,d} {tp_:>14,d} {tp_*B_ZIP/1e9:>8.0f}GB {tp_/1500/3600:>10.0f}h")


if __name__ == "__main__":
    main()
