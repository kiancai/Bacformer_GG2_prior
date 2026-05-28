#!/bin/bash
# Bacformer GPU 任务通用启动包装：用 caiqy 的 glibc 2.35 + nvidia driver 库 +
# conda env libs 启动 python,绕开 CentOS 7 系统 glibc 2.17 太老 (flash-attn 编译产物
# 需 GLIBC_2.32+,系统 glibc 2.17 不满足).
#
# 用法 (从 MCFProjet 根目录,或任意 cwd):
#   CUDA_VISIBLE_DEVICES=1 bash bacformer_prior/scripts/_run_with_glibc.sh \
#       bacformer_prior/scripts/3.embed.py --K-max 32 --shard 0/2 --gpu 0
#
# 等价于直接跑:
#   /home/cml_lab/anaconda3/envs/caiqy_bacformer_prior/bin/python \
#       bacformer_prior/scripts/3.embed.py --K-max 32 --shard 0/2 --gpu 0
# 但 import flash-attn / faesm 时会因 GLIBC_2.32 not found 而崩.
#
# 详细环境复刻过程见 .claude/docs/bacformer_prior/decisions.md §19-§20 +
# experiments.md §8.

set -euo pipefail

# 必要路径 (硬编码到 caiqy 个人路径,如换机器需改)
readonly GLIBC=/home/cml_lab/caiqy/glibc
readonly CONDA=/home/cml_lab/anaconda3/envs/caiqy_bacformer_prior
readonly NVIDIA=/usr/local/nvidia/lib

# 验路径都存在
for p in "$GLIBC/ld-linux-x86-64.so.2" "$GLIBC/libc.so.6" "$CONDA/bin/python" "$NVIDIA/libcuda.so.1"; do
    [ -f "$p" ] || [ -L "$p" ] || { echo "ERROR: $p 不存在,环境失败" >&2; exit 1; }
done

# library-path 顺序: caiqy glibc 在最前 (覆盖系统 2.17) → conda lib (torch CUDA .so) →
# nvidia lib (driver libcuda) → 系统 /usr/lib64 兜底 (libpython 等不被 glibc 覆盖的)
exec "$GLIBC/ld-linux-x86-64.so.2" \
    --library-path "$GLIBC:$CONDA/lib:$CONDA/lib64:$NVIDIA:/usr/lib64" \
    "$CONDA/bin/python" "$@"
