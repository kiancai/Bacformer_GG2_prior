# scripts — bacformer_prior CLI 入口

沿用 MCFProjet 子项目约定：**编号脚本 = 1 对 1 CLI wrapper，每个脚本 = 一个步骤**。

| 脚本 | 步骤 | 算力 | 状态 | 产出（→ `data/bacformer_prior/`） |
|---|---|---|---|---|
| `0.audit_coverage.py` | 全 8114 词表对 GTDB r220 覆盖率审计（read-only） | CPU | ✅ | `coverage_audit.tsv`（含 K_g / match / 盲区） |
| `0b.blindspot_audit.py` | 盲区菌刻画 + phylo 借力可行性（read-only） | CPU | ✅ | stdout（Domain/门/丰度/借力距离） |
| `1.build_mapping.py` | genus → GTDB r220 各 species 代表基因组 | CPU | ✅ | `genus_to_genomes.tsv`（98,255 行 / **62,373 唯一基因组**） |
| `2.download_faa.py` | 取 NCBI 现成蛋白 `.faa`（**用户手动跑**；断点续/并行；无注释的记 missing） | IO | ✅ | `faa/<acc>.faa.gz`（40,870 落） + `logs/{missing_protein,accessions,download_errors}.txt` |
| `2b.prodigal.py` | 对 `missing_protein.txt` (~21.5k 个) 下 `.fna` + **pyrodigal** 现做蛋白 | CPU+IO | 🔄 跑中（用户手动） | `faa/<acc>.faa.gz`（追加 ~21k） + `logs/prodigal_{no_fna,zero,errors}.txt` |
| `_build_quality_cache.py` | （内部辅助）从 GTDB metadata 抽 `quality_score = completeness − 5×contam` | CPU | ✅ | `acc_quality.tsv`（供 3/4 选种） |
| `3.embed.py` | **small** Bacformer 前向（480 维），每基因组一向量；按 K_max cap | **GPU** | 🟡 骨架（待装 `bacformer`+ESM-2 填实 forward） | `genome_embeddings/<acc>.npy`（每基因组 480d fp32） |
| `4.pack_tensor.py` | 聚合/padding `(V, K_max, 480)` + mask + fallback 补盲区；同时落 `species` + `mean` 两版 | CPU | ✅ 骨架（dry-run 跑通） | `genus_prior.species.{npz,pt}` + `genus_prior.mean.{npz,pt}` + `pack_report.txt` |

## 算力 / 依赖

- 仅 `3.embed.py` 需 GPU。`0/0b/1/_build_quality_cache` 可在 GPU 被占时先跑。
- 起长任务（下载 / GPU 前向）前，先在 `.claude/docs/ACTIVE_WORK.md` 看板登记。
- **env**：
  - `MiCoFormerV2`（项目主 env）—— 跑 0/0b/1/2/_build_quality_cache/4
  - **`caiqy_bacformer_prior`**（独立 env, python 3.11 + pyrodigal 3.7.1）—— 跑 2b；后续装 `bacformer`+`torch`+ESM-2 后也跑 3
- Bacformer 主路用 **small**（480 维，ESM-2 t12 35M 底座）；large（960 维 / ESM-C 300M，GPU ~10×）留作升级版备选。

## 命令速查

```bash
# 从 MCFProjet 根目录跑（共用账户 conda run 解析错 python，用 env 绝对路径）

# 已跑完
/home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python bacformer_prior/scripts/0.audit_coverage.py
/home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python bacformer_prior/scripts/0b.blindspot_audit.py
/home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python bacformer_prior/scripts/1.build_mapping.py
/home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python bacformer_prior/scripts/2.download_faa.py
/home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python bacformer_prior/scripts/_build_quality_cache.py

# 跑中（2b prodigal）
/home/cml_lab/anaconda3/envs/caiqy_bacformer_prior/bin/python bacformer_prior/scripts/2b.prodigal.py --workers 8

# 骨架就绪（等 bacformer 装包 + GPU 上）
/home/cml_lab/anaconda3/envs/caiqy_bacformer_prior/bin/python bacformer_prior/scripts/3.embed.py --limit 100 --gpu 0     # pilot
/home/cml_lab/anaconda3/envs/caiqy_bacformer_prior/bin/python bacformer_prior/scripts/3.embed.py --K-max 32 --gpu 0      # 全量

# 骨架就绪（dry-run 已通过；待 3 完成）
/home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python bacformer_prior/scripts/4.pack_tensor.py --K-max 32 --fallback-threshold 6.0
```

完整 CLI 参数详解 → `.claude/docs/bacformer_prior/commands.md`。
