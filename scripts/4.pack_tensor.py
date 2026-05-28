"""4. 聚合 + padding + fallback → 最终张量 (V=8114, K_max, 480) + mask + offline mean。

承接 3.embed 的 `genome_embeddings/<acc>.npy`，按属聚合：
  - 每属取 top-K_max 个基因组（按 quality 排序）padding 到 (K_max, 480) + mask
  - 397 个盲区按 fallback 规则补：同 Family + patristic<阈值 借力 / 否则 zero+mask
  - 同时落两份：
    * genus_prior.species.{npz,pt}  → (V, K_max, 480) + mask，给 learned attention pool
    * genus_prior.mean.{npz,pt}     → (V, 480) offline mean pool，资源默认版 + 消融对照

骨架阶段（2026-05-28）：完整实现；可 dry-run（无 embed npy 时仍能跑过 fallback 计算）。

用法（从 MCFProjet 根目录）：
    python bacformer_prior/scripts/4.pack_tensor.py --K-max 32 --fallback-threshold 6.0
    python bacformer_prior/scripts/4.pack_tensor.py --K-max 16 --output-formats npz  # 只落 npz
    python bacformer_prior/scripts/4.pack_tensor.py --dry-run  # 不读 npy，只验 fallback 决策

输出：
    data/bacformer_prior/genus_prior.species.{npz,pt}
    data/bacformer_prior/genus_prior.mean.{npz,pt}
    data/bacformer_prior/pack_report.txt  （每属实际填充统计 + fallback 决策审计）

env: caiqy_bacformer_prior（python 3.11 + numpy；--output-formats pt 时需 torch）
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Iterable

import numpy as np

ROOT = "/home/cml_lab/caiqy/project/MCFProjet"
DATA = f"{ROOT}/data/bacformer_prior"
EMB_DIR = f"{DATA}/genome_embeddings"
MAPPING = f"{DATA}/genus_to_genomes.tsv"
QUALITY = f"{DATA}/acc_quality.tsv"
AUDIT = f"{DATA}/coverage_audit.tsv"
H5AD = f"{ROOT}/data/gg2/MCFCorpusV2.gg2.h5ad"

D = 480  # Bacformer small embedding 维度（large 升级时改 960）


# ──────────────────────────────────────────────────────────────────────
# Load inputs
# ──────────────────────────────────────────────────────────────────────

def load_mapping_quality_audit():
    """返回 (token_to_accs sorted by quality desc, token_match_type, V_real)。"""
    # token → [acc] 按 quality 排序
    quality: dict[str, float] = {}
    with open(QUALITY) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            quality[row["accession"]] = float(row["quality_score"])

    token_to_accs: dict[int, list[str]] = defaultdict(list)
    with open(MAPPING) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            token_to_accs[int(row["token_idx"])].append(row["accession"])
    for t in token_to_accs:
        token_to_accs[t] = sorted(set(token_to_accs[t]), key=lambda a: -quality.get(a, -1e9))

    # audit: token → match_type / K_g
    match_type: dict[int, str] = {}
    with open(AUDIT) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            match_type[int(row["token_idx"])] = row["match_type"]
    V_real = max(match_type.keys()) + 1  # 假设连续 0..V-1

    return token_to_accs, match_type, V_real


def load_phylo_and_taxonomy(V_real: int):
    """读 h5ad 拿 phylo_dist + Family/Order（fallback 用）。返回 (phylo (V,V), family list, order list)。"""
    import anndata as ad
    a = ad.read_h5ad(H5AD, backed="r")
    phylo = np.asarray(a.varp["phylo_dist"]).astype(np.float32)
    family = a.var["Family"].astype(str).values
    order = a.var["Order"].astype(str).values
    assert phylo.shape == (V_real, V_real), f"phylo shape {phylo.shape} != ({V_real},{V_real})"
    a.file.close()
    return phylo, family, order


def load_embeddings(accs_to_load: set[str], dry_run: bool = False) -> dict[str, np.ndarray]:
    """读 genome_embeddings/*.npy 进 dict。dry_run 时返回空 dict（不读 npy）。"""
    if dry_run:
        return {}
    embeddings: dict[str, np.ndarray] = {}
    for acc in accs_to_load:
        path = f"{EMB_DIR}/{acc}.npy"
        if os.path.exists(path):
            vec = np.load(path)
            assert vec.shape == (D,), f"vec shape {vec.shape} != ({D},)"
            embeddings[acc] = vec.astype(np.float32)
    return embeddings


# ──────────────────────────────────────────────────────────────────────
# Fallback decision
# ──────────────────────────────────────────────────────────────────────

def decide_fallback(
    blind_tokens: list[int],
    has_genome_tokens: list[int],
    phylo: np.ndarray,
    family: np.ndarray,
    order: np.ndarray,
    patristic_threshold: float,
) -> list[tuple[int, int | None, str]]:
    """对每个盲区 token 决定 fallback：借哪个邻居 token / 还是 mask=0。

    规则：同 Family + patristic<threshold → 借；否则 mask=0。

    Returns: [(blind_token, source_token_or_None, decision)]
        decision ∈ {'borrow_same_family', 'mask_far_family', 'mask_diff_family'}
    """
    has_arr = np.array(has_genome_tokens)
    decisions = []
    for bi in blind_tokens:
        dists = phylo[bi, has_arr]
        j = int(np.argmin(dists))
        nearest = int(has_arr[j])
        d = float(dists[j])
        same_family = (family[bi] == family[nearest]) and family[bi] != ""
        if same_family and d < patristic_threshold:
            decisions.append((bi, nearest, "borrow_same_family"))
        elif same_family:
            decisions.append((bi, None, "mask_far_family"))  # 同 F 但 patristic 太远
        else:
            decisions.append((bi, None, "mask_diff_family"))  # 不同 F
    return decisions


# ──────────────────────────────────────────────────────────────────────
# Pack tensor
# ──────────────────────────────────────────────────────────────────────

def pack(
    token_to_accs: dict[int, list[str]],
    embeddings: dict[str, np.ndarray],
    V_real: int,
    K_max: int,
    fallback_decisions: list[tuple[int, int | None, str]],
    dry_run: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """生成 (species_tensor, species_mask, mean_tensor, mean_mask, stats)。

    species_tensor: (V_real, K_max, D)
    species_mask:   (V_real, K_max) bool
    mean_tensor:    (V_real, D)
    mean_mask:      (V_real,) bool（False 表示无任何真实 species 也无 fallback 借力）
    """
    if dry_run:
        species_tensor = np.zeros((V_real, K_max, D), dtype=np.float32)
        species_mask = np.zeros((V_real, K_max), dtype=bool)
    else:
        species_tensor = np.zeros((V_real, K_max, D), dtype=np.float32)
        species_mask = np.zeros((V_real, K_max), dtype=bool)

    stats = {
        "n_real_filled": 0,        # 有真实 species 填充的 token 数
        "n_truncated": 0,           # K_g > K_max 被截的 token 数
        "n_blind_borrowed": 0,
        "n_blind_masked": 0,
        "n_missing_embeddings": 0,  # acc 在 mapping 但 npy 没拿到
        "species_used": defaultdict(int),  # 每属实际填了几个槽
    }

    # ── 真实 token 填充 ──
    if not dry_run:
        for t in range(V_real):
            accs = token_to_accs.get(t, [])
            if not accs:
                continue
            filled = 0
            if len(accs) > K_max:
                stats["n_truncated"] += 1
            for acc in accs[:K_max]:
                if acc not in embeddings:
                    stats["n_missing_embeddings"] += 1
                    continue
                species_tensor[t, filled] = embeddings[acc]
                species_mask[t, filled] = True
                filled += 1
            stats["species_used"][filled] += 1
            if filled > 0:
                stats["n_real_filled"] += 1

    # ── 盲区 fallback ──
    for blind, source, decision in fallback_decisions:
        if decision == "borrow_same_family" and source is not None:
            if not dry_run and species_mask[source].any():
                # 借源 token 的 species 块（同一份 (K_max, D) 复制）
                species_tensor[blind] = species_tensor[source]
                species_mask[blind] = species_mask[source]
                stats["n_blind_borrowed"] += 1
            elif dry_run:
                stats["n_blind_borrowed"] += 1
        else:
            stats["n_blind_masked"] += 1

    # ── offline mean ──
    mean_tensor = np.zeros((V_real, D), dtype=np.float32)
    mean_mask = np.zeros(V_real, dtype=bool)
    if not dry_run:
        cnt = species_mask.sum(axis=1)  # (V_real,)
        valid = cnt > 0
        sums = species_tensor.sum(axis=1)  # (V_real, D)
        mean_tensor[valid] = sums[valid] / cnt[valid, None]
        mean_mask[:] = valid

    stats["species_used"] = dict(stats["species_used"])
    return species_tensor, species_mask, mean_tensor, mean_mask, stats


# ──────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────

def save_outputs(
    species_tensor: np.ndarray,
    species_mask: np.ndarray,
    mean_tensor: np.ndarray,
    mean_mask: np.ndarray,
    K_max: int,
    fallback_threshold: float,
    version: str,
    formats: list[str],
) -> None:
    base = f"{DATA}/genus_prior"
    meta = {
        "K_max": K_max,
        "version": version,
        "fallback_strategy": "phylo_borrow_same_family",
        "fallback_threshold": fallback_threshold,
        "D": D,
    }

    if "npz" in formats:
        np.savez_compressed(
            f"{base}.species.npz",
            embeddings=species_tensor,
            mask=species_mask,
            **{f"meta_{k}": np.array([v]) for k, v in meta.items()},
        )
        np.savez_compressed(
            f"{base}.mean.npz",
            embeddings=mean_tensor,
            mask=mean_mask,
            **{f"meta_{k}": np.array([v]) for k, v in meta.items()},
        )
        print(f"  wrote {base}.species.npz + {base}.mean.npz")

    if "pt" in formats:
        import torch
        torch.save({
            "embeddings": torch.from_numpy(species_tensor),
            "mask": torch.from_numpy(species_mask),
            "meta": meta,
        }, f"{base}.species.pt")
        torch.save({
            "embeddings": torch.from_numpy(mean_tensor),
            "mask": torch.from_numpy(mean_mask),
            "meta": meta,
        }, f"{base}.mean.pt")
        print(f"  wrote {base}.species.pt + {base}.mean.pt")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K-max", type=int, default=32, help="每属 species 槽位数（决策门 3.1，默认 32）")
    ap.add_argument("--fallback-threshold", type=float, default=6.0,
                    help="同 Family + patristic<X 才借（决策门 4.1，默认 6.0；experiments.md §7.2 推荐）")
    ap.add_argument("--output-formats", nargs="+", default=["npz", "pt"], choices=["npz", "pt"])
    ap.add_argument("--version", default="small_v1_r220_gg2_2409", help="产物 version 字符串")
    ap.add_argument("--dry-run", action="store_true", help="不读 npy、不写文件，仅算 fallback 决策 + 输出 stats")
    args = ap.parse_args()

    print(f"=== load mapping + quality + audit ===", flush=True)
    token_to_accs, match_type, V_real = load_mapping_quality_audit()
    blind_tokens = [t for t, m in match_type.items() if m == "none"]
    has_genome_tokens = [t for t, m in match_type.items() if m != "none"]
    print(f"  V_real={V_real}, has_genome={len(has_genome_tokens)}, blind={len(blind_tokens)}", flush=True)

    print(f"=== load phylo + taxonomy ===", flush=True)
    phylo, family, order = load_phylo_and_taxonomy(V_real)

    print(f"=== decide fallback (K_max={args.K_max}, patristic<{args.fallback_threshold}) ===", flush=True)
    fallback_decisions = decide_fallback(
        blind_tokens, has_genome_tokens, phylo, family, order, args.fallback_threshold,
    )
    n_borrow = sum(1 for _, _, d in fallback_decisions if d == "borrow_same_family")
    n_mask_f = sum(1 for _, _, d in fallback_decisions if d == "mask_far_family")
    n_mask_d = sum(1 for _, _, d in fallback_decisions if d == "mask_diff_family")
    print(f"  借力: {n_borrow}, mask 同F远: {n_mask_f}, mask 异F: {n_mask_d}", flush=True)

    print(f"=== load embeddings ===", flush=True)
    if args.dry_run:
        print(f"  --dry-run: 跳过 npy 加载", flush=True)
        embeddings = {}
    else:
        # 收集要加载的 acc
        needed_accs = set()
        for t, accs in token_to_accs.items():
            needed_accs.update(accs[: args.K_max])
        embeddings = load_embeddings(needed_accs)
        print(f"  loaded {len(embeddings):,} / {len(needed_accs):,} embeddings", flush=True)
        if len(embeddings) < len(needed_accs):
            print(f"  WARN: {len(needed_accs) - len(embeddings):,} 个 acc 没找到 npy", flush=True)

    print(f"=== pack tensor ===", flush=True)
    species_tensor, species_mask, mean_tensor, mean_mask, stats = pack(
        token_to_accs, embeddings, V_real, args.K_max, fallback_decisions, args.dry_run,
    )
    print(f"  species_tensor shape={species_tensor.shape} ({species_tensor.nbytes/1e9:.2f}GB)")
    print(f"  species_mask    shape={species_mask.shape}    ({species_mask.nbytes/1e9:.2f}GB)")
    print(f"  mean_tensor     shape={mean_tensor.shape}     ({mean_tensor.nbytes/1e6:.1f}MB)")
    print(f"  stats: {stats}")

    if not args.dry_run:
        print(f"=== save ({','.join(args.output_formats)}) ===", flush=True)
        save_outputs(
            species_tensor, species_mask, mean_tensor, mean_mask,
            args.K_max, args.fallback_threshold, args.version, args.output_formats,
        )
        # 写报告
        with open(f"{DATA}/pack_report.txt", "w") as f:
            f.write(f"K_max={args.K_max}\nfallback_threshold={args.fallback_threshold}\nversion={args.version}\n\n")
            f.write(f"V_real={V_real}\n")
            f.write(f"has_genome_tokens={len(has_genome_tokens)}\n")
            f.write(f"blind_tokens={len(blind_tokens)}\n")
            f.write(f"  borrowed (same Family + pat<{args.fallback_threshold}): {n_borrow}\n")
            f.write(f"  masked (same Family far): {n_mask_f}\n")
            f.write(f"  masked (diff Family):     {n_mask_d}\n\n")
            f.write(f"pack stats: {stats}\n")
            f.write(f"mean_mask.sum() = {mean_mask.sum()} / {V_real} ({100*mean_mask.sum()/V_real:.2f}%)\n")
        print(f"  wrote {DATA}/pack_report.txt")

    print("\n完成 ✓")


if __name__ == "__main__":
    main()
