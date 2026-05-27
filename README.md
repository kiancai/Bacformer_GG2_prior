# bacformer_prior

为 **GG2 24.09 genus 词表**构建基于 **Bacformer**（基因组基础模型）的 genus 级 dense embedding —
既是一个独立、可复用、可发布的**资源**，也是 MiCoFormer 的一个候选**结构先验**。

> 完整设计、动机、与 MiCoFormer roadmap §4.7 的关系、决策记录：见
> **`MCFProjet/.claude/docs/bacformer_prior/design.md`**（本 README 只给一句话概览 + 目录约定）。

## 两个产物

1. **资源**：覆盖 GG2 24.09 全 genus 词表的 `genus -> Bacformer genome embedding`。
   只要用 GG2 24.09 注释，任意 genus 都能查到向量（含基因组缺失菌的 fallback）。
   成败看**覆盖率 + 质量 + 可复现**，与模型涨不涨点无关。
2. **模型组件**：可学聚合层（species 代表基因组 -> genus 向量，end-to-end 训）+ 注入
   （镜像 `MiCoFormer/micoformer/models/phylo_pe.py`）。落在 **MiCoFormer `dev` 分支**，不在本仓库。

## 数据链路（产物一）

```
GG2 24.09 genus 词表
   └─(1 mapping)→ GTDB r220 该 genus 下各 species 的代表基因组 accession
        └─(2 download)→ NCBI datasets CLI 取 .faa（蛋白序列，无 GPU，IO 密集）
             └─(3 embed)→ Bacformer-large 前向，每基因组一个 dense 向量（需 GPU）
                  └─(4 pack)→ (V, K_max, dim) + mask 张量 + 每基因组缓存（产物一交付物）
                       └─(5 inject)→ MiCoFormer 端可学聚合 + 加性注入（产物二）
```

## 目录约定

```
bacformer_prior/            ← 本子项目（独立 git；根 git 不跟踪，见根 .gitignore）
├── bacformer_prior/        ← python 包：mapping / download / embed / aggregate
├── scripts/                ← 编号 CLI 入口（见 scripts/README.md）
├── pyproject.toml          ← editable 安装（pip install -e .）
└── requirements.txt        ← 依赖（安装需用户许可）
```

- **大文件产物（.faa / embedding 缓存 / 打包张量）** → `MCFProjet/data/bacformer_prior/`（根级 gitignored），**不进本仓库**。
- **GTDB r220 metadata / 映射表**等中等中间文件同样进 `data/bacformer_prior/`。

## 状态

🟡 **脚手架阶段**（仅目录 + 文档 + git）。尚未下载任何数据、未占用 GPU。
下一步从 read-only 的**覆盖率审计**起（全 8114 词表对 GTDB r220 能拿到几成基因组），见 design.md。
