"""3. Bacformer small 前向：每基因组一个 480 维 dense vec（GPU，需装 bacformer+ESM-2）。

⚠️ **骨架阶段**（2026-05-28）：
  数据流 / CLI / IO / 断点续 已完成；
  Bacformer forward 部分先 `_load_bacformer_model()` placeholder（NotImplementedError），
  待装 `bacformer` + `transformers` + ESM-2 后填实。

承接：2 download + 2b prodigal 后 `faa/<acc>.faa.gz`（~62,200 个）。
按 K_max cap：每属按 quality 取 top-K_max 个基因组进 embed。pilot 后定 K_max（决策门 3.1）。

用法（从 MCFProjet 根目录）：
    # pilot 测吞吐 + 决 K_max
    python bacformer_prior/scripts/3.embed.py --limit 100 --gpu 0

    # 全量（按 K_max 决定的子集）
    python bacformer_prior/scripts/3.embed.py --K-max 32 --gpu 0

    # 测某属
    python bacformer_prior/scripts/3.embed.py --token-ids 100,200,300 --gpu 0

输出：
    data/bacformer_prior/genome_embeddings/<acc>.npy  每基因组一个 (480,) fp32 向量

env: caiqy_bacformer_prior（python 3.11 + pyrodigal + 装包后 + bacformer/torch/transformers）
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import time
from collections import defaultdict
from typing import Iterable

import numpy as np

ROOT = "/home/cml_lab/caiqy/project/MCFProjet"
DATA = f"{ROOT}/data/bacformer_prior"
FAA_DIR = f"{DATA}/faa"
EMB_DIR = f"{DATA}/genome_embeddings"
MAPPING = f"{DATA}/genus_to_genomes.tsv"
QUALITY = f"{DATA}/acc_quality.tsv"
LOG_DIR = f"{DATA}/logs"

# Bacformer small 常数（与 architecture.md §1 / data.md §1.4 一致）
BACFORMER_MODEL = "macwiatrak/bacformer-masked-complete-genomes"
HIDDEN_SIZE = 480
MAX_PROTEINS = 6000


# ──────────────────────────────────────────────────────────────────────
# Input loading
# ──────────────────────────────────────────────────────────────────────

def load_mapping_and_quality() -> tuple[dict[int, list[str]], dict[str, float]]:
    """读 genus_to_genomes.tsv 聚合 token→[acc] + acc_quality.tsv 给质量分。"""
    token_to_accs: dict[int, list[str]] = defaultdict(list)
    with open(MAPPING) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            token_to_accs[int(row["token_idx"])].append(row["accession"])
    # 去重 + 按 quality 排序
    quality: dict[str, float] = {}
    with open(QUALITY) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            quality[row["accession"]] = float(row["quality_score"])
    for t, accs in token_to_accs.items():
        token_to_accs[t] = sorted(set(accs), key=lambda a: -quality.get(a, -1e9))
    return token_to_accs, quality


def select_accs_for_embed(
    token_to_accs: dict[int, list[str]],
    K_max: int,
    token_filter: set[int] | None = None,
) -> list[str]:
    """按 K_max cap 取每属 top-K_max 个 acc，返回去重后的总 acc 集合。"""
    accs = set()
    for t, ts_accs in token_to_accs.items():
        if token_filter is not None and t not in token_filter:
            continue
        accs.update(ts_accs[:K_max] if K_max > 0 else ts_accs)
    # 进一步只保留 faa/ 已落盘的（漏抓的 no_fna/zero 自然跳过）
    accs = {a for a in accs if os.path.exists(f"{FAA_DIR}/{a}.faa.gz")}
    return sorted(accs)


# ──────────────────────────────────────────────────────────────────────
# FASTA reading + ordering
# ──────────────────────────────────────────────────────────────────────

def read_proteins_ordered(faa_path: str, max_proteins: int = MAX_PROTEINS) -> list[str]:
    """读 .faa.gz 蛋白序列，按 locus 顺序排（既支持 NCBI 注释 也支持 prodigal）。

    返回:list[str] 蛋白序列（不含 FASTA header），最多 max_proteins 个（超截断）。
    """
    proteins: list[tuple[str, str]] = []  # [(header_id, seq), ...]
    cur_id, cur_seq = None, []
    with gzip.open(faa_path, "rt") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id is not None:
                    proteins.append((cur_id, "".join(cur_seq)))
                # prodigal header: '>contig_id_N # start # end # ...'
                # NCBI header: '>WP_XXX.X ...'
                cur_id = line[1:].split(maxsplit=1)[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        proteins.append((cur_id, "".join(cur_seq)))

    # 按 header_id 排序（prodigal 的 contig_id_N 自然按 contig + 序号；NCBI WP_ 通常已按 locus）
    proteins.sort(key=lambda p: p[0])
    # 截断到 max_proteins
    if len(proteins) > max_proteins:
        proteins = proteins[:max_proteins]
    return [seq for _, seq in proteins]


# ──────────────────────────────────────────────────────────────────────
# Bacformer model wrapper (placeholder until bacformer 装包)
# ──────────────────────────────────────────────────────────────────────

def _load_bacformer_model(device: str):
    """加载 Bacformer small + ESM-2 t12 35M 底座，返回 embed_one(proteins) callable。

    用法对齐 HF model card 标准 example:
    - AutoModel + trust_remote_code=True
    - bfloat16 推理
    - protein_seqs_to_bacformer_inputs 内部已经把蛋白经 ESM-2 编成 480d;
      输出 shape (B, L_with_special, 480),L_with_special = N_proteins + ~3 special tokens
    - last_hidden_state.mean(dim=1) 取 genome embedding (HF 标准)

    ⚠️ 环境约束 (2026-05-28 实测,代价惨痛):
    必须 transformers >= 4.45, < 5（4.x 末版,实测 4.57.6 OK）。
    transformers 5.x major upgrade 改了 from_pretrained 流程,
    persistent=False buffer 在加载后会被重分配成 empty 内存,
    覆盖 BacformerEncoder.__init__ 里 precompute_freqs_cis 算好的 freqs_cos/sin
    → 出未初始化垃圾值或 NaN → forward 全 NaN。
    若用了 transformers 5.x, env 即坏, 必须降回 4.x。详 requirements.txt 注释。
    """
    import torch  # 局部 import 避免影响 dry-run 路径
    from bacformer.pp import protein_seqs_to_bacformer_inputs
    from transformers import AutoModel

    model = (
        AutoModel.from_pretrained(BACFORMER_MODEL, trust_remote_code=True)
        .to(device)
        .eval()
        .to(torch.bfloat16)
    )

    def embed_one(proteins: list[str]) -> np.ndarray:
        # protein_seqs_to_bacformer_inputs 内部用 ESM-2 底座把每蛋白编成 480d
        # （内部 batch_size 控制 ESM-2 前向的 batch；与 Bacformer 自身 batch 无关）
        inputs = protein_seqs_to_bacformer_inputs(
            proteins, device=device, batch_size=128, max_n_proteins=MAX_PROTEINS,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, return_dict=True)
        # last_hidden_state: (1, L_with_special, 480) → mean → (1, 480) → (480,) fp32
        vec = out["last_hidden_state"].mean(dim=1).squeeze(0).float().cpu().numpy().astype(np.float32)
        return vec

    return embed_one


# ──────────────────────────────────────────────────────────────────────
# Main embedding loop
# ──────────────────────────────────────────────────────────────────────

def embed_one_acc(acc: str, embed_fn, max_proteins: int = MAX_PROTEINS) -> tuple[str, int, np.ndarray | None]:
    """嵌入单个 acc。返回 (state, n_proteins, vec or None)。"""
    out_path = f"{EMB_DIR}/{acc}.npy"
    if os.path.exists(out_path):
        return "skip", 0, None
    faa_path = f"{FAA_DIR}/{acc}.faa.gz"
    if not os.path.exists(faa_path):
        return "no_faa", 0, None
    proteins = read_proteins_ordered(faa_path, max_proteins)
    if not proteins:
        return "zero_prot", 0, None
    try:
        vec = embed_fn(proteins)
    except Exception as e:
        print(f"  ERROR {acc}: {type(e).__name__}: {e}", flush=True)
        return "error", len(proteins), None
    assert vec.shape == (HIDDEN_SIZE,), f"vec shape {vec.shape} != ({HIDDEN_SIZE},)"
    # np.save 会自动加 .npy 后缀；给一个不含 .npy 的临时基名,落盘后改 rename
    tmp_base = out_path[:-4] + ".tmp"  # 'X.npy' -> 'X.tmp', np.save 会加 .npy
    np.save(tmp_base, vec)
    os.replace(tmp_base + ".npy", out_path)
    return "ok", len(proteins), vec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K-max", type=int, default=32, help="每属保留 species 数（决策门 3.1，默认 32）")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index（CUDA_VISIBLE_DEVICES）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 个 acc（pilot）")
    ap.add_argument("--token-ids", type=str, default="", help="逗号分隔 token_idx，仅跑这些 token 的 acc（debug）")
    ap.add_argument("--max-proteins", type=int, default=MAX_PROTEINS, help="单基因组蛋白上限（默认 6000）")
    ap.add_argument("--dry-run", action="store_true", help="不加载模型、不写文件，只列待跑 acc + 估算时长")
    args = ap.parse_args()

    os.makedirs(EMB_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 限定 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ── 选 acc ──
    token_to_accs, _quality = load_mapping_and_quality()
    token_filter = None
    if args.token_ids:
        token_filter = {int(x) for x in args.token_ids.split(",") if x.strip()}
        print(f"token filter: {len(token_filter)} tokens", flush=True)
    accs = select_accs_for_embed(token_to_accs, args.K_max, token_filter)
    if args.limit:
        accs = accs[: args.limit]
    print(f"待 embed {len(accs):,} 个 acc（K_max={args.K_max}）", flush=True)

    if args.dry_run:
        # 估算总蛋白 + 时长
        from collections import Counter
        n_proteins_per_acc = []
        for a in accs[:100]:  # 抽样 100
            faa = f"{FAA_DIR}/{a}.faa.gz"
            if not os.path.exists(faa):
                continue
            n_proteins_per_acc.append(sum(1 for line in gzip.open(faa, "rt") if line.startswith(">")))
        if n_proteins_per_acc:
            avg = sum(n_proteins_per_acc) / len(n_proteins_per_acc)
            total = avg * len(accs)
            print(f"  抽样 {len(n_proteins_per_acc)} 个 acc 估蛋白平均 {avg:.0f}/基因组")
            print(f"  总蛋白估算 {total/1e6:.1f}M")
            print(f"  @ 1500 prot/s 估 GPU 时 {total/1500/3600:.1f}h")
            print(f"  @ 800 prot/s（保守）估 {total/800/3600:.1f}h")
        return

    # ── 加载模型 ──
    print(f"loading Bacformer small（GPU {args.gpu}）...", flush=True)
    embed_fn = _load_bacformer_model(device=f"cuda:0")  # CUDA_VISIBLE_DEVICES 已设
    print("loaded ✓", flush=True)

    # ── 主循环 ──
    counts = {"ok": 0, "skip": 0, "no_faa": 0, "zero_prot": 0, "error": 0}
    errors: list[str] = []
    t0 = time.time()
    total_proteins = 0
    for i, acc in enumerate(accs, 1):
        st, npx, _ = embed_one_acc(acc, embed_fn, args.max_proteins)
        counts[st] += 1
        total_proteins += npx
        if st == "error":
            errors.append(acc)
        if i % 50 == 0 or i == len(accs):
            el = time.time() - t0
            rate_acc = i / max(el, 1e-6)
            rate_prot = total_proteins / max(el, 1e-6)
            eta = (len(accs) - i) / max(rate_acc, 1e-6)
            print(f"  [{i:,}/{len(accs):,} {i/len(accs)*100:4.1f}%] "
                  f"ok={counts['ok']} skip={counts['skip']} no_faa={counts['no_faa']} "
                  f"zero={counts['zero_prot']} error={counts['error']} | "
                  f"{rate_acc:.2f} acc/s {rate_prot:.0f} prot/s "
                  f"已用{el/60:.0f}min ETA{eta/60:.0f}min", flush=True)

    # 落 log
    with open(f"{LOG_DIR}/embed_errors.txt", "w") as f:
        f.write("\n".join(errors) + ("\n" if errors else ""))
    print(f"\n完成：{counts}")
    print(f"  total proteins embedded ≈ {total_proteins:,}")
    print(f"  errors → logs/embed_errors.txt（{len(errors)} 个）")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
