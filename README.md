# bacformer_prior

为 **Greengenes2 24.09 全 genus 词表**构建基于 [**Bacformer**](https://github.com/macwiatrak/Bacformer)（基因组基础模型）的 genus 级 dense embedding —
既是一个独立、可复用、可发布的**资源**，也是 [MiCoFormer](https://github.com/kiancai/MiCoFormer) 的一个候选**结构先验**。

> 完整设计、动机、决策记录在主项目 [`MCFProjet/.claude/docs/bacformer_prior/design.md`](https://github.com/kiancai/MCFProjet)；本 README 给概览 + 复现入口。

## 两个产物

1. **资源**：覆盖 GG2 24.09 **全 genus 词表（8,114 个）**的 `genus -> Bacformer genome embedding`。
   只要用 GG2 24.09 注释，**任意** genus（含无基因组的 genus，走 fallback）都能查到向量、直接喂模型。
   成败看**覆盖率 + 质量 + 可复现**，与下游模型涨不涨点无关。
2. **模型组件**：可学聚合层（species 代表基因组 → genus 向量，end-to-end 训）+ 注入。
   落在 [MiCoFormer `dev` 分支](https://github.com/kiancai/MiCoFormer)，**不在本仓库**。

## 数据链路（产物一）

```
GG2 24.09 genus 词表 (8,114)
   └─(0 audit)→  全词表 vs GTDB r220 覆盖率（read-only）
                 7,717 命中 (95.1%) / 397 盲区
   └─(1 mapping)→ genus → GTDB r220 各 species 代表基因组 accession
                 98,255 (token×species) → 62,373 唯一代表基因组
                 RefSeq 24,106 (38.7%) / GenBank 38,267 (61.3%)
   └─(2 download)→ NCBI Datasets API 取 .faa（蛋白序列，IO 密集，无 GPU）
                 40,870 直接拿到 / 21,503 NCBI 无现成注释 → 走 (2b)
   └─(2b prodigal)→ 对无注释 acc 下 .fna + Prodigal 预测蛋白（CPU）
   └─(3 embed)→  Bacformer small 前向（480 维 / ESM-2 t12 35M 底座），每基因组一向量（需 GPU）
   └─(4 pack)→   (V, K_max, 480) + mask 张量 + 盲区 fallback（产物一交付物）
   └─(5 inject)→ MiCoFormer 端可学 attention pool + 加性注入（产物二）
```

## 数据源

- **词表**：[Greengenes2 24.09](https://greengenes2.ucsd.edu/) — 8,114 genus 16S 参考词表。
- **基因组**：[GTDB r220](https://gtdb.ecogenomic.org/) — GG2 24.09 基于的同 release，taxonomy 零漂移（match_type 全 exact、base 回退 = 0）。
- **基因组源 .faa / .fna**：[NCBI Datasets API v2alpha](https://www.ncbi.nlm.nih.gov/datasets/)。
- **Bacformer 模型**：[`macwiatrak/bacformer-masked-complete-genomes`](https://huggingface.co/macwiatrak/bacformer-masked-complete-genomes)（small；ESM-2 t12 35M 底座；480 维；6,000 蛋白上限）。

## 目录约定

```
bacformer_prior/                ← 本仓库（独立 git）
├── bacformer_prior/            ← python 包（mapping / download / embed / aggregate）
├── scripts/                    ← 编号 CLI 入口（见 scripts/README.md）
│   ├── 0.audit_coverage.py     全词表 vs GTDB r220 覆盖率审计（read-only）
│   ├── 0b.blindspot_audit.py   盲区刻画 + phylo 借力可行性（read-only）
│   ├── 1.build_mapping.py      genus → GTDB r220 species 代表
│   ├── 2.download_faa.py       NCBI 蛋白 .faa 下载（手动，断点续）
│   ├── 2b.prodigal.py          无注释 acc 走 .fna + Prodigal（待添加）
│   ├── 3.embed.py              Bacformer 前向，每基因组一向量（待添加）
│   └── 4.pack_tensor.py        聚合 padding 成最终张量（待添加）
├── pyproject.toml              editable 安装
└── requirements.txt            依赖（Bacformer 包安装方式待 3.embed 启动时定）
```

- **大文件产物**（.faa / .fna / embedding 缓存 / 打包张量）→ `MCFProjet/data/bacformer_prior/`（根级 gitignored），**不进本仓库**。
- 中等中间文件（GTDB metadata、mapping tsv）同样进 `data/bacformer_prior/`。

## 当前状态

| 步 | 状态 | 关键产物 |
|---|---|---|
| 0 audit / 0b blindspot | ✅ 完成 | 95.1% 覆盖、397 盲区刻画、96% 可 phylo 借 |
| 1 mapping | ✅ 完成 | `genus_to_genomes.tsv`：98,255 行 / **62,373 唯一基因组** |
| **2 download** | ✅ 完成（**40,870 .faa.gz 落盘**） | `faa/<acc>.faa.gz` + `logs/missing_protein.txt` (21,503) |
| 2b prodigal | ⬜ 待添加 | — |
| 3 embed | ⬜ 待添加 | — |
| 4 pack | ⬜ 待添加 | — |

## 复现

```bash
# 安装（CPU 步骤够用；3.embed 启动时再装 bacformer + torch）
pip install -e .

# 准备 GTDB r220 参考（taxonomy + metadata，~150MB）
# 详见 scripts/README.md

# 0–4：从 MCFProjet 根目录跑
python bacformer_prior/scripts/0.audit_coverage.py
python bacformer_prior/scripts/1.build_mapping.py
python bacformer_prior/scripts/2.download_faa.py    # IO 密集；断点续；几小时
# ... 2b / 3 / 4 见 scripts/README.md
```

## 引用

- **Bacformer**: Wiatrak et al., *Bacformer: protein language model for bacterial complete-genome representation*, HuggingFace `macwiatrak/bacformer-masked-complete-genomes`.
- **GTDB r220**: Parks et al., *GTDB: an ongoing census of bacterial and archaeal diversity through a phylogenetically consistent, rank normalized and complete genome-based taxonomy*, NAR 2022. Release 220 (2024).
- **Greengenes2 2024.09**: McDonald et al., *Greengenes2 unifies microbial data in a single reference tree*, Nat Biotechnol 2023.

## License

MIT — 见 [LICENSE](LICENSE)。本仓库代码与中间产物本身遵循 MIT；衍生使用的上游模型 / 数据库（Bacformer、GTDB、GG2、NCBI）请遵循各自原始许可。
