"""0. 覆盖率审计（read-only）：GG2 24.09 全 genus 词表 vs GTDB r220。

回答："完备"能做到几成 —— 词表里每个 genus 在 GTDB r220 下能拿到几个 species 代表基因组，
盲区（无任何基因组的 genus）有多大。不下载基因组、不占 GPU。

输入：
  - 词表：data/gg2/MCFCorpusV2.gg2.h5ad 的 var['Genus']（8114, g__X）、var['observed']
  - GTDB r220 taxonomy：data/bacformer_prior/refs/{bac120,ar53}_taxonomy_r220.tsv.gz
    （每行 = 一个 species 代表基因组 accession <TAB> d__;...;g__X;s__Y）

输出：data/bacformer_prior/coverage_audit.tsv（每 genus：match 类型 / K_g / 是否有基因组 / observed）
     + stdout 汇总（覆盖率、K_g 分布、下载量估算、盲区）。

命名对齐（沿用 20260526_genus_function_ko 的处理）：
  - GG2 genus 带结尾 `_<数字>` 节点号（Moraxella_C_651924）→ normalize 去掉。
  - GG2 24.09 与 GTDB r220 同 release，多系字母码（Moraxella_C）应直接对上；base 回退仅作兜底。
"""
import gzip
import re
import sys
from collections import defaultdict

import anndata as ad
import numpy as np

ROOT = "/home/cml_lab/caiqy/project/MCFProjet"
H5AD = f"{ROOT}/data/gg2/MCFCorpusV2.gg2.h5ad"
REFS = f"{ROOT}/data/bacformer_prior/refs"
TAX_FILES = [f"{REFS}/bac120_taxonomy_r220.tsv.gz", f"{REFS}/ar53_taxonomy_r220.tsv.gz"]
OUT = f"{ROOT}/data/bacformer_prior/coverage_audit.tsv"


def strip_prefix(x: str) -> str:
    return x[3:] if x[:3] in ("g__", "s__") else x


def normalize(g: str) -> str:
    # 去 GG2 结尾节点号 _<数字>（可叠加）
    return re.sub(r"(_\d+)+$", "", g)


def base_name(g: str) -> str:
    # 再去 GTDB 多系字母码 _<大写>（仅兜底用）
    return re.sub(r"(_[A-Z])+$", "", normalize(g))


def main() -> None:
    # ---- 词表 ----
    a = ad.read_h5ad(H5AD, backed="r")
    gg2_genus = a.var["Genus"].astype(str).values  # g__X
    observed = a.var["observed"].astype(bool).values if "observed" in a.var else np.zeros(len(gg2_genus), bool)
    a.file.close()
    V = len(gg2_genus)
    print(f"词表 genus 总数 = {V}（observed={int(observed.sum())}）")

    # ---- GTDB r220：genus -> {species}, 及每 species 一个代表 accession ----
    genus_species = defaultdict(set)
    base_species = defaultdict(set)
    n_rows = 0
    for path in TAX_FILES:
        with gzip.open(path, "rt") as f:
            for line in f:
                acc, tax = line.rstrip("\n").split("\t", 1)
                n_rows += 1
                parts = tax.split(";")
                if len(parts) < 7:
                    continue
                g = strip_prefix(parts[5])
                s = strip_prefix(parts[6])
                if not g or not s:
                    continue
                gn = normalize(g)
                genus_species[gn].add(s)
                base_species[base_name(g)].add(s)
    print(f"GTDB r220 species 代表基因组行数 = {n_rows:,}; distinct genus = {len(genus_species):,}")

    # ---- 逐 genus 匹配 ----
    rows = []
    k_exact = k_base = k_none = 0
    for i, g_raw in enumerate(gg2_genus):
        gn = normalize(strip_prefix(g_raw))
        if gn in genus_species:
            kg, mt = len(genus_species[gn]), "exact"
            k_exact += 1
        elif base_name(g_raw) in base_species:
            kg, mt = len(base_species[base_name(g_raw)]), "base"
            k_base += 1
        else:
            kg, mt = 0, "none"
            k_none += 1
        rows.append((i, g_raw, gn, mt, kg, bool(observed[i])))

    # ---- 写出 ----
    with open(OUT, "w") as fo:
        fo.write("token_idx\tgg2_genus\tnormalized\tmatch_type\tK_g\tobserved\n")
        for i, g_raw, gn, mt, kg, obs in rows:
            fo.write(f"{i}\t{g_raw}\t{gn}\t{mt}\t{kg}\t{obs}\n")

    # ---- 汇总 ----
    kg_all = np.array([r[4] for r in rows])
    has = kg_all > 0
    obs_mask = np.array([r[5] for r in rows])

    def cov(mask):
        return 100.0 * (has & mask).sum() / max(mask.sum(), 1)

    print("\n=== 覆盖率（有 ≥1 基因组的 genus 占比）===")
    print(f"  全词表        : {cov(np.ones(V, bool)):.1f}%  ({int(has.sum())}/{V})")
    print(f"  observed 子集 : {cov(obs_mask):.1f}%  ({int((has & obs_mask).sum())}/{int(obs_mask.sum())})")
    print(f"  匹配类型: exact={k_exact}, base={k_base}, none(盲区)={k_none}")

    vals = kg_all[has]
    print("\n=== K_g 分布（有基因组的 genus）===")
    for q in (50, 75, 90, 95, 99, 100):
        print(f"  p{q:<3d} = {np.percentile(vals, q):.0f}")
    print(f"  mean = {vals.mean():.1f}")

    print("\n=== 下载量估算（每 species 一个代表）===")
    print(f"  全下（不设 cap）= {int(vals.sum()):,} 个基因组")
    for k in (8, 16, 32, 64):
        capped = np.minimum(kg_all, k).sum()
        notrunc = 100.0 * (vals <= k).mean()
        print(f"  K_max={k:>3d}: 下载≈{int(capped):,}, 不截断 genus 占比 {notrunc:.0f}%, 张量(全词表,480d) {V*k*480*4/1e9:.2f}GB")

    print("\n=== K_g 最大 12（多为大属，需 cap+子采）===")
    top = sorted(rows, key=lambda r: -r[4])[:12]
    for i, g_raw, gn, mt, kg, obs in top:
        print(f"  {kg:>5d}  {g_raw}  ({mt}{', obs' if obs else ''})")

    # 盲区里 observed 的（这些是真正会被样本用到、却无基因组的菌）
    blind_obs = [r for r in rows if r[4] == 0 and r[5]]
    print(f"\n=== 盲区中 observed 的 genus 数 = {len(blind_obs)}（这些最需要 fallback）===")
    for r in blind_obs[:15]:
        print(f"  {r[1]}")
    if len(blind_obs) > 15:
        print(f"  ...（共 {len(blind_obs)} 个）")

    print(f"\n写出: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
