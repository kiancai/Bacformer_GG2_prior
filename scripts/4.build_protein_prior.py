"""4. 构建 genus 级 protein prior：聚合特征与 centered cosine 距离矩阵。

承接 3.embed 的 genome_embeddings/<acc>.npy，按属：
  1. quality 加权聚合 K 个基因组 embedding → genus 特征 (V_real, 480)
  2. 盲区（397 无基因组）fallback：同 Family + patristic<阈值 借邻居特征
  3. 全局 centering（破各向异性 cone；raw 随机对 cosine 0.96 挤一团，centered 后才有区分度）
  4. centered cosine 距离矩阵 (V_real, V_real) → 喂 MiCoFormer protein Tree-W（phylo_w 镜像）

产物（data/bacformer_prior/protein_prior/）：
  protein_dist.npy   (V_real,V_real) f32  ← Tree-W 距离矩阵（主产物，d = 1 - centered_cosine ∈ [0,2]）
  protein_feat.npy   (V_real,480)    f32  ← centered genus 特征（可选输入先验）
  valid_mask.npy     (V_real,)       bool ← True=有真实/借来的特征；False=完全盲区
  meta.json          K_max/weighting/metric/dist_scale/n_valid/n_borrowed/n_invalid/...

注意：valid_mask 是资源审计产物；当前 MiCoFormer 消费者不会自动加载它。

用法（MCFProjet 根目录）：
  python data/bacformer_prior/_src/scripts/4.build_protein_prior.py --K-max 32 --weighting quality
  python data/bacformer_prior/_src/scripts/4.build_protein_prior.py --half-check
  OMP_NUM_THREADS=4 ... 前缀防 BLAS 线程 oversubscribe

env: caiqy_bacformer_prior 或 MiCoFormerV2（numpy + anndata）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

ROOT = "/home/cml_lab/caiqy/project/MCFProjet"
DATA = f"{ROOT}/data/bacformer_prior"
EMB_DIR = f"{DATA}/genome_embeddings"
MAPPING = f"{DATA}/genus_to_genomes.tsv"
QUALITY = f"{DATA}/acc_quality.tsv"
AUDIT = f"{DATA}/coverage_audit.tsv"
H5AD = f"{ROOT}/data/gg2/MCFCorpusV2.gg2.h5ad"

D = 480  # Bacformer small embedding 维度


# ──────────────────────────────────────────────────────────────────────
# Load inputs
# ──────────────────────────────────────────────────────────────────────

def load_mapping_quality_audit():
    """返回 (token_to_accs 按 quality desc 排序, quality dict, match_type, V_real)。"""
    quality: dict[str, float] = {}
    with open(QUALITY) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            quality[row["accession"]] = float(row["quality_score"])

    token_to_accs: dict[int, list[str]] = defaultdict(list)
    with open(MAPPING) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            token_to_accs[int(row["token_idx"])].append(row["accession"])
    for t in token_to_accs:
        # quality desc，accession 名做 tiebreaker（去重 + 确定性）
        token_to_accs[t] = sorted(set(token_to_accs[t]),
                                  key=lambda a: (-quality.get(a, -1e9), a))

    match_type: dict[int, str] = {}
    with open(AUDIT) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            match_type[int(row["token_idx"])] = row["match_type"]
    V_real = max(match_type.keys()) + 1
    return token_to_accs, quality, match_type, V_real


def load_phylo_and_taxonomy(V_real: int):
    """读 h5ad 拿 phylo_dist + Family（fallback 用）。返回 (phylo (V,V), family list)。"""
    import anndata as ad
    a = ad.read_h5ad(H5AD, backed="r")
    phylo = np.asarray(a.varp["phylo_dist"]).astype(np.float32)
    family = a.var["Family"].astype(str).values
    assert phylo.shape == (V_real, V_real), f"phylo {phylo.shape} != ({V_real},{V_real})"
    a.file.close()
    return phylo, family


def load_embeddings(accs_to_load: set[str]) -> dict[str, np.ndarray]:
    """读 genome_embeddings/*.npy（只读已落盘的，missing 自然跳过）。"""
    emb: dict[str, np.ndarray] = {}
    for acc in accs_to_load:
        p = f"{EMB_DIR}/{acc}.npy"
        if os.path.exists(p):
            v = np.load(p)
            assert v.shape == (D,), f"{acc} shape {v.shape}"
            emb[acc] = v.astype(np.float32)
    return emb


def decide_fallback(blind_tokens, has_genome_tokens, phylo, family, threshold):
    """盲区 token → 借哪个邻居 / mask。规则：同 Family + patristic<threshold → 借。"""
    has_arr = np.array(has_genome_tokens)
    decisions = []
    for bi in blind_tokens:
        dists = phylo[bi, has_arr]
        j = int(np.argmin(dists))
        nearest = int(has_arr[j])
        d = float(dists[j])
        same_family = (family[bi] == family[nearest]) and family[bi] != ""
        if same_family and d < threshold:
            decisions.append((bi, nearest, "borrow_same_family"))
        else:
            decisions.append((bi, None, "mask"))
    return decisions


# ──────────────────────────────────────────────────────────────────────
# Aggregate + centering + distance matrix
# ──────────────────────────────────────────────────────────────────────

def aggregate_genus_features(token_to_accs, embeddings, quality, V_real, K_max, weighting):
    """每属 quality 加权（或等权）聚合 top-K 基因组 embedding → genus 特征。

    Returns: feat (V_real, D) float32, has_feat (V_real,) bool, n_used (V_real,) int
    """
    feat = np.zeros((V_real, D), dtype=np.float32)
    has_feat = np.zeros(V_real, dtype=bool)
    n_used = np.zeros(V_real, dtype=int)
    for t in range(V_real):
        accs = [a for a in token_to_accs.get(t, [])[:K_max] if a in embeddings]
        if not accs:
            continue
        vs = np.stack([embeddings[a] for a in accs])  # (k, D)
        if weighting == "quality" and len(accs) > 1:
            q = np.array([quality.get(a, 0.0) for a in accs], dtype=np.float64)
            w = q - q.min() + 1e-3          # shift 非负（quality_score = comp-5*contam 可能为负）
            w = w / w.sum()
            f = (w[:, None] * vs).sum(0)
        else:
            f = vs.mean(0)
        feat[t] = f.astype(np.float32)
        has_feat[t] = True
        n_used[t] = len(accs)
    return feat, has_feat, n_used


def apply_fallback(feat, has_feat, fallback_decisions):
    """盲区借同 Family 邻居的 genus 特征。返回 borrow 计数。"""
    n_borrow = 0
    borrowed_flag = np.zeros(len(has_feat), dtype=bool)
    for blind, source, decision in fallback_decisions:
        if decision == "borrow_same_family" and source is not None and has_feat[source]:
            feat[blind] = feat[source]
            has_feat[blind] = True
            borrowed_flag[blind] = True
            n_borrow += 1
    return n_borrow, borrowed_flag


def build_distance_matrix(feat, has_feat):
    """全局 centering（仅用 valid 行估均值）→ centered cosine 距离矩阵 (V,V)。

    盲区行列设中性距离（valid 非对角距离的中位数），靠 valid_mask 让 Tree-W 端可跳过。
    Returns: dist (V,V) f32, feat_centered (V,D) f32, dist_scale float, neutral float
    """
    V = feat.shape[0]
    mu = feat[has_feat].mean(0)                          # 全局均值（破各向异性 cone）
    feat_c = feat - mu
    feat_c[~has_feat] = 0.0                              # 盲区置 0（不参与）

    # L2 normalize（仅 valid 行有意义）
    norm = np.linalg.norm(feat_c, axis=1, keepdims=True)
    unit = feat_c / np.maximum(norm, 1e-8)
    cos = unit @ unit.T                                  # (V,V) centered cosine
    dist = (1.0 - cos).astype(np.float32)                # [0,2]

    # valid 子块的非对角距离统计（dist_scale + 中性值）
    vidx = np.where(has_feat)[0]
    sub = dist[np.ix_(vidx, vidx)]
    off = sub[~np.eye(len(vidx), dtype=bool)]
    dist_scale = float(off.mean())
    neutral = float(np.median(off))

    # 盲区行列设中性（避免误导 Tree-W 的 E[d]），对角线 0
    inv = ~has_feat
    dist[inv, :] = neutral
    dist[:, inv] = neutral
    np.fill_diagonal(dist, 0.0)
    return dist, feat_c.astype(np.float32), dist_scale, neutral


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K-max", type=int, default=32, help="每属聚合 top-K 基因组（按 quality）")
    ap.add_argument("--weighting", choices=["quality", "mean"], default="quality",
                    help="quality 加权 / 等权 mean（默认 quality）")
    ap.add_argument("--fallback-threshold", type=float, default=6.0,
                    help="盲区借同 Family 的 patristic 上限（默认 6.0）")
    ap.add_argument("--out-dir", default="protein_prior", help="输出子目录名")
    ap.add_argument("--half-check", action="store_true",
                    help="用现有(可能不全)embedding 验证逻辑，允许大量 missing")
    args = ap.parse_args()

    print("=== load mapping + quality + audit ===", flush=True)
    token_to_accs, quality, match_type, V_real = load_mapping_quality_audit()
    blind = [t for t, m in match_type.items() if m == "none"]
    has_genome = [t for t, m in match_type.items() if m != "none"]
    print(f"  V_real={V_real}, has_genome={len(has_genome)}, blind={len(blind)}", flush=True)

    print("=== load phylo + taxonomy ===", flush=True)
    phylo, family = load_phylo_and_taxonomy(V_real)

    print("=== load embeddings ===", flush=True)
    needed = set()
    for t, accs in token_to_accs.items():
        needed.update(accs[: args.K_max])
    emb = load_embeddings(needed)
    print(f"  loaded {len(emb):,} / {len(needed):,} embeddings", flush=True)
    if len(emb) < len(needed) and not args.half_check:
        print(f"  WARN: {len(needed)-len(emb):,} 个 acc 无 npy（embed 未跑完？非 --half-check 慎用）", flush=True)

    print("=== aggregate genus features ===", flush=True)
    feat, has_feat, n_used = aggregate_genus_features(
        token_to_accs, emb, quality, V_real, args.K_max, args.weighting)
    print(f"  有真实特征的 genus: {has_feat.sum()} / {V_real}", flush=True)

    print("=== fallback 盲区借同 Family ===", flush=True)
    fb = decide_fallback(blind, has_genome, phylo, family, args.fallback_threshold)
    n_borrow, borrowed_flag = apply_fallback(feat, has_feat, fb)
    n_invalid = int((~has_feat).sum())
    print(f"  借力 {n_borrow} / 盲区总 {len(blind)} → 仍完全盲区(invalid): {n_invalid}", flush=True)

    print("=== build centered cosine distance matrix ===", flush=True)
    dist, feat_c, dist_scale, neutral = build_distance_matrix(feat, has_feat)
    print(f"  protein_dist {dist.shape} ({dist.nbytes/1e9:.2f}GB), "
          f"dist_scale(非零均值)={dist_scale:.4f}, neutral(中位)={neutral:.4f}", flush=True)

    # ── 落盘 ──
    out = f"{DATA}/{args.out_dir}"
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/protein_dist.npy", dist)
    np.save(f"{out}/protein_feat.npy", feat_c)
    np.save(f"{out}/valid_mask.npy", has_feat)
    meta = {
        "V_real": int(V_real), "D": D, "K_max": args.K_max,
        "weighting": args.weighting, "metric": "centered_cosine_distance",
        "fallback_threshold": args.fallback_threshold,
        "n_real_feat": int(has_feat.sum() - n_borrow), "n_borrowed": int(n_borrow),
        "n_invalid": int(n_invalid), "dist_scale": dist_scale, "neutral_dist": neutral,
        "half_check": args.half_check, "n_emb_loaded": len(emb), "n_emb_needed": len(needed),
    }
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"=== wrote {out}/{{protein_dist,protein_feat,valid_mask}}.npy + meta.json ===")
    print(json.dumps(meta, indent=2))
    print("\n完成 ✓")


if __name__ == "__main__":
    main()
