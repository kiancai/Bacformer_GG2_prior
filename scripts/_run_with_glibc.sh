#!/bin/bash
# Bacformer GPU 任务通用启动包装：用 caiqy 的 glibc 2.35 + nvidia driver 库 +
# conda env libs 启动 python,绕开 CentOS 7 系统 glibc 2.17 太老 (flash-attn 编译产物
# 需 GLIBC_2.32+,系统 glibc 2.17 不满足).
#
# 用法 (从 MCFProjet 根目录,或任意 cwd):
#   CUDA_VISIBLE_DEVICES=1 bash data/bacformer_prior/_src/scripts/_run_with_glibc.sh \
#       data/bacformer_prior/_src/scripts/3.embed.py --K-max 32 --shard 0/2
#
# 等价于直接跑:
#   /home/cml_lab/anaconda3/envs/caiqy_bacformer_prior/bin/python \
#       data/bacformer_prior/_src/scripts/3.embed.py --K-max 32 --shard 0/2
# 但 import flash-attn / faesm 时会因 GLIBC_2.32 not found 而崩.
#
# 当前环境与命令边界见 MCFProjet
# .agent/data/resources/bacformer_prior/{commands,provenance}.md。

set -euo pipefail

# 必要路径 (硬编码到 caiqy 个人路径,如换机器需改)
readonly GLIBC=/home/cml_lab/caiqy/glibc
readonly CONDA=/home/cml_lab/anaconda3/envs/caiqy_bacformer_prior
readonly NVIDIA=/usr/local/nvidia/lib

# ⚠️ 限制 CPU 线程,避免 thread oversubscribe (memory feedback_thread_oversubscribe)
# cluster sbatch 申请 24 核;torch/numpy/MKL/OpenBLAS 默认各开 24 thread → 互相 thrashing
# 实测 (单卡 limit=20):
#   默认: 271 prot/s, GPU sm 8% peak  ← thrash
#   OMP=4: 514 prot/s
#   OMP=8: 523 prot/s ← 略优
#   OMP=12/16: 502/494 prot/s (退化)
# 双卡同时跑差异更小 (CPU 不再是瓶颈,GPU forward 自身限速):
#   OMP=4 双卡: 1078 prot/s 总
#   OMP=8 双卡: 1085 prot/s 总 ← 微优
#   OMP=12 双卡: 1065 prot/s 总
# 取 OMP=8: 单卡略优,双卡微优,双卡占 16/24 核(留 8 核给系统+其他)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}

# 验路径都存在
for p in "$GLIBC/ld-linux-x86-64.so.2" "$GLIBC/libc.so.6" "$CONDA/bin/python" "$NVIDIA/libcuda.so.1"; do
    [ -f "$p" ] || [ -L "$p" ] || { echo "ERROR: $p 不存在,环境失败" >&2; exit 1; }
done

# library-path 顺序: caiqy glibc 在最前 (覆盖系统 2.17) → conda lib (torch CUDA .so) →
# nvidia lib (driver libcuda) → 系统 /usr/lib64 兜底 (libpython 等不被 glibc 覆盖的)
exec "$GLIBC/ld-linux-x86-64.so.2" \
    --library-path "$GLIBC:$CONDA/lib:$CONDA/lib64:$NVIDIA:/usr/lib64" \
    "$CONDA/bin/python" "$@"
