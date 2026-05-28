"""辅助：从 GTDB r220 metadata 抽 quality 字段，缓存为 tsv 供 3/4 步用。

输入: data/bacformer_prior/refs/{bac120,ar53}_metadata_r220.tsv.gz
输出: data/bacformer_prior/acc_quality.tsv
     字段: accession / completeness / contamination / protein_count / quality_score

quality_score = completeness - 5×contamination（标准 MAG 质量打分，CheckM2 风格）。
本表用于:
- 3.embed.py 在 K_max cap 时选 top-K_max（按 quality 排序，超 cap 截）
- 4.pack_tensor.py 同样按 quality 给 species 排位（决定 padding 顺序）

仅依赖标准库 + numpy（MiCoFormerV2 env）。一次性，几秒。
"""
import csv
import gzip
import sys

ROOT = "/home/cml_lab/caiqy/project/MCFProjet"
META = [f"{ROOT}/data/bacformer_prior/refs/bac120_metadata_r220.tsv.gz",
        f"{ROOT}/data/bacformer_prior/refs/ar53_metadata_r220.tsv.gz"]
MAPPING = f"{ROOT}/data/bacformer_prior/genus_to_genomes.tsv"
OUT = f"{ROOT}/data/bacformer_prior/acc_quality.tsv"


def main() -> None:
    # 只缓存 mapping 里出现的 62,373 个 acc（其他基因组与本项目无关，省 IO）
    wanted = set()
    with open(MAPPING) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            wanted.add(row["accession"])
    print(f"target accessions: {len(wanted):,}", flush=True)

    # 列索引（实测）：0 accession / 18 gtdb_representative / 19 gtdb_taxonomy
    #                 2 checkm2_completeness / 3 checkm2_contamination / 91 protein_count
    out_rows = []
    seen = set()
    for path in META:
        with gzip.open(path, "rt") as f:
            header = f.readline().rstrip("\n").split("\t")
            # 验证列名（防 GTDB schema 变化）
            assert header[0] == "accession", f"col 0 not accession: {header[0]}"
            assert "completeness" in header[2].lower(), f"col 2 not completeness: {header[2]}"
            assert "contamination" in header[3].lower(), f"col 3 not contamination: {header[3]}"
            assert header[18] == "gtdb_representative", f"col 18 not gtdb_representative: {header[18]}"
            assert "protein_count" in header[91].lower(), f"col 91 not protein_count: {header[91]}"
            for line in f:
                c = line.rstrip("\n").split("\t")
                if c[18] != "t":  # 只取 species representative
                    continue
                if c[0] not in wanted:
                    continue
                seen.add(c[0])
                try:
                    comp = float(c[2]) if c[2] else float("nan")
                    cont = float(c[3]) if c[3] else float("nan")
                except ValueError:
                    comp = cont = float("nan")
                try:
                    nprot = int(c[91]) if c[91] and c[91] != "none" else 0
                except ValueError:
                    nprot = 0
                # quality_score: nan 安全降级到 0（最低优先级）
                if comp != comp or cont != cont:  # NaN check
                    qscore = -1e9
                else:
                    qscore = comp - 5.0 * cont
                out_rows.append((c[0], comp, cont, nprot, qscore))

    print(f"matched: {len(out_rows):,}/{len(wanted):,}", flush=True)
    missing = wanted - seen
    if missing:
        print(f"WARN: {len(missing):,} acc 在 metadata 里找不到（理论上不应出现）；前 3: {list(missing)[:3]}",
              flush=True)

    # 按 quality 降序写出（方便目视 + 下游直接用顺序）
    out_rows.sort(key=lambda r: -r[4])
    with open(OUT, "w") as f:
        f.write("accession\tcompleteness\tcontamination\tprotein_count\tquality_score\n")
        for r in out_rows:
            f.write(f"{r[0]}\t{r[1]:.2f}\t{r[2]:.2f}\t{r[3]}\t{r[4]:.2f}\n")
    print(f"wrote {OUT}", flush=True)

    # 速读统计
    import numpy as np
    qs = np.array([r[4] for r in out_rows if r[4] > -1e8])
    print(f"\nquality_score 分布（n={len(qs):,}, 已剔除 NaN）:")
    for q in (1, 5, 25, 50, 75, 95, 99):
        print(f"  p{q:>3d} = {np.percentile(qs, q):.1f}")
    print(f"  mean = {qs.mean():.1f}, max = {qs.max():.1f}")
    print(f"  quality >=80: {(qs>=80).sum():,} ({(qs>=80).mean()*100:.1f}%)")
    print(f"  quality <50:  {(qs<50).sum():,} ({(qs<50).mean()*100:.1f}%) ← 低质量基因组")


if __name__ == "__main__":
    sys.exit(main())
