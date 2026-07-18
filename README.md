# bacformer_prior

本仓库是 MCFProjet 中 BacFormer protein prior 的**生产源码仓**。大文件输入、中间缓存和最终数组位于
父目录 `data/bacformer_prior/`，不进入本 Git repo。

在 MCFProjet 中，资源身份、provenance、schema、消费者和安全命令的活动文档统一位于：

```text
.agent/data/resources/bacformer_prior/
```

本 README 只说明源码入口和当前交付状态，不另建独立 roadmap 或实验台账。

## 当前交付物

当前正式输出是 `data/bacformer_prior/protein_prior/`：

| 文件 | shape | 作用 |
|---|---:|---|
| `protein_feat.npy` | 8,114 x 480 | centered genus feature；Protein PE 与 `func_bacformer` sample view |
| `protein_dist.npy` | 8,114 x 8,114 | centered cosine distance；Protein-W |
| `valid_mask.npy` | 8,114 | real/borrowed feature 有效性 |
| `meta.json` | JSON | K、weighting、fallback 与覆盖统计 |

2026-07-15 只读审计结果：7,717 个 genus 有真实 feature，204 个通过同 Family phylogenetic fallback
借用，193 个仍 invalid；最终数组均 finite，distance 对称且对角为 0。

这些结果证明当前资产内部一致，但最终 `meta.json` 没有绑定 producer commit、输入 hash、模型 revision、
环境和实际命令。因此当前资源的可重建性是 `partial`，不能把本仓库当前 HEAD 直接当成已证明的 build
revision。

## 当前生产链

```text
GG2 2024.09 genus index + GTDB r220 taxonomy/metadata
  -> 0.audit_coverage.py
  -> 1.build_mapping.py + _build_quality_cache.py
  -> 2.download_faa.py
  -> 2b.prodigal.py for accessions without PROT_FASTA
  -> 3.embed.py: Bacformer small, one 480d vector per genome
  -> 4.build_protein_prior.py: quality aggregation + fallback + centered cosine distance
  -> protein_prior/{protein_feat,protein_dist,valid_mask}.npy + meta.json
```

`4.pack_tensor.py` 是早期 padding tensor 原型，不是当前正式输出路线。当前消费者读取
`4.build_protein_prior.py` 生成的 `protein_feat.npy` / `protein_dist.npy`。

## 源码目录

```text
_src/
├── bacformer_prior/               # 轻量 Python package
├── scripts/
│   ├── 0.audit_coverage.py        # GG2 x GTDB coverage
│   ├── 0b.blindspot_audit.py      # blindspot 只读分析
│   ├── 1.build_mapping.py         # genus -> representative genomes
│   ├── _build_quality_cache.py    # GTDB quality cache
│   ├── 2.download_faa.py          # NCBI PROT_FASTA
│   ├── 2b.prodigal.py             # FNA + pyrodigal fallback
│   ├── 3.embed.py                 # Bacformer genome embedding
│   ├── 4.build_protein_prior.py   # 当前最终 prior
│   ├── 4.pack_tensor.py           # 旧原型，不是当前正式产物
│   └── _run_with_glibc.sh         # GPU embedding runtime wrapper
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## 使用与重建边界

- 正常训练只读取现有 prior，不运行本仓库的生产脚本。
- 生产脚本会重写 TSV、下载大量序列、写 embedding 或覆盖最终资源；重建前必须先登记任务、保存旧
  manifest，并核对 MCFProjet 的 `.agent/project/active_work.md`。
- 当前命令和环境边界见 [`scripts/README.md`](scripts/README.md)；更完整的集成说明见 MCFProjet
  `.agent/data/resources/bacformer_prior/commands.md`。
- `pyproject.toml` 目前声明 Python `>=3.11`，而现场可用的 `caiqy_bacformer_prior` 环境是 Python
  3.10.20。这是 packaging 声明与历史运行环境的已知不一致；重新安装前应单独裁定，不在文档整理中
  静默改依赖约束。

## 上游资源与许可

- [Greengenes2 2024.09](https://forum.qiime2.org/t/greengenes2-2024-09/31606)
- [GTDB](https://gtdb.ecogenomic.org/), release r220
- [NCBI Datasets](https://www.ncbi.nlm.nih.gov/datasets/)
- [Bacformer small](https://huggingface.co/macwiatrak/bacformer-masked-complete-genomes)

[LICENSE](LICENSE) 的 MIT 条款只覆盖本仓库代码。GG2、GTDB、NCBI 数据、BacFormer 模型以及由它们
生成的资源分别受上游许可和 data-use 条款约束；不能把代码许可扩展到输入或输出数据。
