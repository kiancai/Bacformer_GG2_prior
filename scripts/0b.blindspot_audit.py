"""0b. 盲区刻画 + phylo 借力可行性（read-only）。

承接 0.audit_coverage：审计出的"无基因组 genus"（match_type=none）到底是什么菌、有多重要、
以及能不能按进化距离从"有基因组的近邻"借/合并 —— 给 fallback 规则提供证据（不下结论）。

输入：
  - data/gg2/MCFCorpusV2.gg2.h5ad：var 全谱系(Domain..Genus) + mass_fraction + observed；varp['phylo_dist']
  - data/bacformer_prior/coverage_audit.tsv：match_type（none=盲区）

输出：stdout 汇总（Domain/门分布、名字形态、丰度权重、借力距离、最近邻共享秩、Top-by-mass）。
结论（2026-05-27 实测）：盲区 397（observed 299）仅占 0.4% mass；~96% 有同科/目的有基因组近邻 → 借力对绝大多数合理。
"""
import re
import numpy as np
import pandas as pd
import anndata as ad

ROOT = "/home/cml_lab/caiqy/project/MCFProjet"
H5AD = f"{ROOT}/data/gg2/MCFCorpusV2.gg2.h5ad"
AUDIT = f"{ROOT}/data/bacformer_prior/coverage_audit.tsv"


def base(g: str) -> str:
    g = g[3:] if g.startswith("g__") else g
    return re.sub(r"(_[A-Z])+$", "", re.sub(r"(_\d+)+$", "", g))


def is_placeholder(g: str) -> bool:
    b = base(g)
    return bool(re.search(r"\d", b)) or bool(re.match(r"^[A-Z0-9-]+$", b)) or "-" in b


def main() -> None:
    a = ad.read_h5ad(H5AD, backed="r")
    cols = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "observed", "mass_fraction"]
    var = a.var[cols].reset_index(drop=True)
    phylo = a.varp["phylo_dist"][:]  # (V, V) float32
    a.file.close()
    V = len(var)

    aud = pd.read_csv(AUDIT, sep="\t")
    blind = aud["match_type"].values == "none"
    having = ~blind
    obs = var["observed"].values
    mf = var["mass_fraction"].values
    print(f"全词表 {V}; 盲区 {blind.sum()}（observed {(blind & obs).sum()}）; 有基因组 {having.sum()}")

    bv = var[blind]
    print("\n=== 盲区 Domain ===\n" + bv["Domain"].value_counts().to_string())
    print("\n=== 盲区 Top Phylum ===\n" + bv["Phylum"].value_counts().head(10).to_string())
    ph = bv["Genus"].map(is_placeholder)
    print(f"\n名字形态: 占位码 {int(ph.sum())} / 拉丁名 {int((~ph).sum())}")
    print(f"\n丰度: 盲区总 mass {mf[blind].sum()*100:.3f}% (其中 observed {mf[blind & obs].sum()*100:.3f}%)")

    # ---- 借力：每个盲区菌 → 最近的有基因组菌 ----
    hav_idx = np.where(having)[0]
    bli_idx = np.where(blind)[0]
    sub = phylo[np.ix_(bli_idx, hav_idx)]
    nn_global = hav_idx[sub.argmin(axis=1)]
    nn_dist = sub.min(axis=1)
    hh = phylo[np.ix_(hav_idx, hav_idx)].copy()
    np.fill_diagonal(hh, np.inf)
    base_nn = hh.min(axis=1)

    def shared_rank(i, j):
        for r in ["Family", "Order", "Class", "Phylum"]:
            if var[r].values[i] == var[r].values[j]:
                return r
        return "Domain+"

    sr = pd.Series([shared_rank(bli_idx[k], nn_global[k]) for k in range(len(bli_idx))])
    print(f"\n借力距离(盲区→最近有基因组菌): 中位 {np.median(nn_dist):.2f}, p90 {np.percentile(nn_dist,90):.2f}, "
          f"max {nn_dist.max():.2f}  |  基线(有基因组互相最近邻) 中位 {np.median(base_nn):.2f}")
    order = ["Family", "Order", "Class", "Phylum", "Domain+"]
    print("\n=== 最近有基因组菌共享秩（决定借力合理性）===")
    obs_b = obs[bli_idx]
    for r in order:
        n, no = int((sr == r).sum()), int((sr[obs_b] == r).sum())
        print(f"  同{r:<8s}: 全部 {n:4d}({n/len(sr)*100:.0f}%)  observed {no:4d}({no/max(obs_b.sum(),1)*100:.0f}%)")

    print("\n=== Top 12 盲区 by mass ===")
    bdf = pd.DataFrame({
        "genus": var["Genus"].values[bli_idx], "phylum": var["Phylum"].values[bli_idx],
        "obs": obs_b, "mass%": mf[bli_idx] * 100,
        "nn_having": var["Genus"].values[nn_global], "shared": sr.values, "nn_dist": nn_dist,
    }).sort_values("mass%", ascending=False)
    print(bdf.head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
