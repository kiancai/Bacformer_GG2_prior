# BacFormer prior 生产脚本

编号脚本是一条顺序生产链。正常模型训练不需要重跑；下列入口会写
`data/bacformer_prior/`，只在明确的重建任务中执行。

## 当前阶段状态

| 脚本 | 作用 | 算力 | 2026-07-15 在盘状态 |
|---|---|---|---|
| `0.audit_coverage.py` | GG2 8,114 genus 对 GTDB r220 覆盖审计 | CPU | 完成；7,717 exact、397 none |
| `0b.blindspot_audit.py` | blindspot 与 phylogenetic fallback 可行性分析 | CPU | 完成；只输出分析 |
| `1.build_mapping.py` | genus 到 GTDB species representative mapping | CPU | 完成；98,255 行、62,373 unique accession |
| `_build_quality_cache.py` | completeness/contamination/protein-count quality cache | CPU | 完成；62,373 行 |
| `2.download_faa.py` | NCBI PROT_FASTA 下载 | IO | 完成；与 2b 合计 62,177 个 FASTA |
| `2b.prodigal.py` | 无 PROT_FASTA 时下载 FNA 并用 pyrodigal 预测 | CPU+IO | 完成；196 个 accession 无可用 FNA |
| `3.embed.py` | Bacformer small per-genome 480d embedding | GPU | 完成当前需求；44,152 个 embedding |
| `4.build_protein_prior.py` | quality aggregation、fallback、centered cosine distance | CPU | 当前正式路线；已生成 8,114 行 prior |
| `4.pack_tensor.py` | 旧 padding tensor 原型 | CPU | 非当前正式路线；保留追溯 |

## 环境边界

- `MiCoFormerV2`：0/0b/1/2/quality cache/4。
- `caiqy_bacformer_prior`：2b 与 3；现场环境为 Python 3.10.20、pyrodigal 3.7.1，Bacformer
  small + ESM-2 t12 35M。
- `3.embed.py` 依赖本地 glibc wrapper。GPU 卡由 Slurm `GRES IDX` 决定，必须在外部显式设置
  `CUDA_VISIBLE_DEVICES`；脚本没有 `--gpu` 参数。
- 未经用户明确许可不安装或升级依赖。`requirements.txt` 记录历史可工作 pin，不是自动安装指令。

## 只读检查

从 MCFProjet 根目录运行：

```bash
git -C data/bacformer_prior/_src status --short
git -C data/bacformer_prior/_src rev-parse HEAD
sed -n '1,200p' data/bacformer_prior/protein_prior/meta.json

/home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python -c \
  "import numpy as np; p='data/bacformer_prior/protein_prior/'; \
   print(np.load(p+'protein_feat.npy', mmap_mode='r').shape); \
   print(np.load(p+'protein_dist.npy', mmap_mode='r').shape); \
   print(np.load(p+'valid_mask.npy', mmap_mode='r').shape)"
```

## 生产入口

以下只是当前有效入口，不表示应当立即重跑：

```bash
PY=/home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python

# 0/1 没有 argparse；不要用 --help 探测，否则会直接执行并重写 TSV
$PY data/bacformer_prior/_src/scripts/0.audit_coverage.py
$PY data/bacformer_prior/_src/scripts/0b.blindspot_audit.py
$PY data/bacformer_prior/_src/scripts/1.build_mapping.py
$PY data/bacformer_prior/_src/scripts/_build_quality_cache.py

$PY data/bacformer_prior/_src/scripts/2.download_faa.py --workers 8
/home/cml_lab/anaconda3/envs/caiqy_bacformer_prior/bin/python \
  data/bacformer_prior/_src/scripts/2b.prodigal.py --workers 8

# dry-run 不加载模型、不写 embedding
/home/cml_lab/anaconda3/envs/caiqy_bacformer_prior/bin/python \
  data/bacformer_prior/_src/scripts/3.embed.py --K-max 32 --dry-run

# GPU embedding：<allocated-index> 必须与当前 Slurm 分配一致
CUDA_VISIBLE_DEVICES=<allocated-index> \
  bash data/bacformer_prior/_src/scripts/_run_with_glibc.sh \
  data/bacformer_prior/_src/scripts/3.embed.py --K-max 32 --shard 0/1

OMP_NUM_THREADS=4 $PY data/bacformer_prior/_src/scripts/4.build_protein_prior.py \
  --K-max 32 --weighting quality --fallback-threshold 6.0 \
  --out-dir protein_prior
```

## 重建前置条件

1. 在 MCFProjet `.agent/project/active_work.md` 登记任务；GPU 另做 Slurm/CUDA preflight。
2. 保存当前输入、mapping、最终数组和 `meta.json` 的 hash，不覆盖唯一证据。
3. 记录 producer commit、完整命令、环境、GG2 index hash、四个 GTDB hash、模型 revision/hash。
4. 输出 selected genomes/weights、fallback donor/status 与最终文件 hash，形成真正的 build manifest。
5. 完成后按 MCFProjet 数据 registry 和实验回写规则更新活动文档。

集成侧的完整命令说明以 MCFProjet
`.agent/data/resources/bacformer_prior/commands.md` 为准。
