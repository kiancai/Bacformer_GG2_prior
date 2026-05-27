# scripts — bacformer_prior CLI 入口

沿用 MCFProjet 子项目约定：**编号脚本 = 1 对 1 CLI wrapper，每个脚本 = 一个步骤**。
当前为脚手架，下列为**计划入口**（尚未实现，故 scripts/ 仅含本 README）：

| 脚本 | 步骤 | 算力 | 产出（→ `data/bacformer_prior/`） |
|---|---|---|---|
| `0.audit_coverage.py` | 全 8114 词表对 GTDB r220 的覆盖率审计（read-only，先做） | CPU | `coverage_audit.tsv`（每 genus：是否有基因组 / K_g / match 类型 / 盲区标记） |
| `1.build_mapping.py` | genus -> GTDB r220 各 species 代表基因组 accession | CPU | `genus_to_genomes.tsv` |
| `2.download_faa.py` | NCBI datasets CLI 批量取 `.faa.gz` | IO（无 GPU） | `faa/<accession>.faa.gz` |
| `3.embed.py` | Bacformer-large 前向，每基因组 dense 向量 | **GPU** | `genome_embeddings/`（{accession: vec}） |
| `4.pack_tensor.py` | 聚合/padding 成 `(V, K_max, dim) + mask` + fallback 补盲区 | CPU | `genus_prior.{npz,pt}` |

> 算力提醒：仅 `3.embed.py` 需 GPU。`0/1/2` 可在 GPU 被占时先跑（见 `.claude/docs/ACTIVE_WORK.md`）。
> 起任何长任务（下载 / GPU 前向）前，先在 ACTIVE_WORK 看板登记。
