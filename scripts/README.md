# scripts — bacformer_prior CLI 入口

沿用 MCFProjet 子项目约定：**编号脚本 = 1 对 1 CLI wrapper，每个脚本 = 一个步骤**。
| 脚本 | 步骤 | 算力 | 状态 | 产出（→ `data/bacformer_prior/`） |
|---|---|---|---|---|
| `0.audit_coverage.py` | 全 8114 词表对 GTDB r220 覆盖率审计（read-only） | CPU | ✅ | `coverage_audit.tsv`（含 K_g / match / 盲区） |
| `0b.blindspot_audit.py` | 盲区菌刻画 + phylo 借力可行性（read-only） | CPU | ✅ | stdout（Domain/门/丰度/借力距离） |
| `1.build_mapping.py` | genus -> GTDB r220 各 species 代表基因组 | CPU | ✅ | `genus_to_genomes.tsv`（98,255 行 / **62,373 唯一基因组**） |
| `2.download_faa.py` | 取唯一 ~62k 代表蛋白 `.faa`（**用户手动跑**；断点续/并行；无注释的记 missing） | IO | ✅就绪 | `faa/<acc>.faa.gz` + `logs/{missing_protein,accessions}.txt` |
| `2b.prodigal.py` | 对 `missing_protein.txt`（无注释 GenBank，~28%）下 `.fna` + Prodigal 预测蛋白 | CPU | 待写（需装 prodigal） | `faa/<acc>.faa.gz`（补全） |
| `3.embed.py` | **small** Bacformer 前向（480 维），每基因组一向量；按 K_max cap | **GPU** | 待写（需装 bacformer+ESM-2） | `genome_embeddings/`（{acc: vec480}） |
| `4.pack_tensor.py` | 聚合/padding 成 `(V, K_max, 480) + mask` + fallback 补盲区 | CPU | 待写 | `genus_prior.{npz,pt}` |

> 算力：仅 `3.embed.py` 需 GPU。`0/1/2/2b` 可在 GPU 被占时先跑（见 `.claude/docs/ACTIVE_WORK.md`）。
> 起长任务（下载 / GPU 前向）前，先在 ACTIVE_WORK 看板登记。
> `2` 现选 **small** 模型（480 维，ESM-2 35M 底座）；large（960 维 /ESM-C 300M，GPU 贵 ~10×）留作日后升级。
