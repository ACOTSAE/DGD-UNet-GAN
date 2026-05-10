#!/bin/bash

set -euo pipefail

# 检查依赖
command -v convert &>/dev/null || { echo "错误: 未找到 convert (ImageMagick)"; exit 1; }
command -v parallel &>/dev/null || { echo "错误: 未找到 parallel (GNU parallel)"; exit 1; }

# 检查参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIR="$1"
if [ ! -d "$DIR" ]; then
    echo "Error: '$DIR' is not a directory."
    exit 1
fi

# 需要转换的扩展名（不包含 png）
EXTENSIONS=(jpg jpeg gif bmp tiff tif webp)

# 构建 find 条件（-o 列表）
find_args=()
for ext in "${EXTENSIONS[@]}"; do
    find_args+=(-o -iname "*.$ext")
done
unset find_args[0]          # 移除第一个多余的 -o

echo "Scanning for non-PNG image files in '$DIR'..."

# 查找并并行转换
find "$DIR" -type f \( "${find_args[@]}" \) -print0 | \
    parallel -0 --no-notice 'convert {} {.}.png && echo "Converted: {} -> {.}.png"'

echo "All done."
