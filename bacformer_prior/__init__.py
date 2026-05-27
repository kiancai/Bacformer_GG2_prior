"""bacformer_prior — GG2 24.09 genus 词表的 Bacformer 基因组 embedding 资源 + MiCoFormer 结构先验。

两个产物（详见 `.claude/docs/bacformer_prior/design.md`）：
  - 产物一（资源）：覆盖 GG2 24.09 全 genus 词表的 "genus -> Bacformer 基因组 embedding"，
    独立、可复用、可发布；成败看覆盖率 + 质量 + 可复现，与下游模型涨不涨点无关。
  - 产物二（模型组件）：可学聚合层 + 注入（镜像 phylo_pe），落在 MiCoFormer `dev` 分支，消费产物一。
"""

__version__ = "0.0.1"
