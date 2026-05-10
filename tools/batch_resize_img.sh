#!/bin/bash
# batch_resize.sh - 批量缩放图片到指定百分比，使用 GNU parallel 加速

set -eo pipefail

command -v convert &>/dev/null 2>&1 || { echo "错误: 未找到 convert (ImageMagick)"; exit 1; }
command -v parallel &>/dev/null 2>&1 || { echo "错误: 未找到 parallel (GNU parallel)"; exit 1; }

usage() {
    cat <<EOF
用法: $0 -s <源目录> -d <目标目录> -p <百分比> [-r] [-h]

选项:
  -s <目录>   源目录，包含要缩放的图片
  -d <目录>   目标目录，缩放后的图片保存到此
  -p <数字>   缩放百分比（如 70 表示 70%）
  -r          递归处理源目录的子目录（默认只处理当前目录）
  -h          显示此帮助信息

依赖: ImageMagick 的 convert 命令, GNU parallel
EOF
}

# 默认值
recursive=false

# 解析参数
while getopts "s:d:p:rh" opt; do
    case "$opt" in
        s) src_dir="$OPTARG" ;;
        d) dest_dir="$OPTARG" ;;
        p) percentage="$OPTARG" ;;
        r) recursive=true ;;
        h) usage ; exit 0 ;;
        *) usage ; exit 1 ;;
    esac
done

# 检查必要参数
if [[ -z "$src_dir" || -z "$dest_dir" || -z "$percentage" ]]; then
    echo "错误: 缺少必要参数 -s, -d, -p"
    usage
fi

# 检查依赖
command -v convert &>/dev/null || { echo "错误: 未找到 convert (ImageMagick)"; exit 1; }
command -v parallel &>/dev/null || { echo "错误: 未找到 parallel (GNU parallel)"; exit 1; }

# 验证源目录
if [[ ! -d "$src_dir" ]]; then
    echo "错误: 源目录 '$src_dir' 不存在"
    exit 1
fi

# 创建目标目录
mkdir -p "$dest_dir" || { echo "错误: 无法创建目标目录 '$dest_dir'"; exit 1; }

# ========== 使用扩展名数组构建 find 条件 ==========
extensions=(jpg jpeg png gif bmp tiff webp)

find_pattern=()
for ext in "${extensions[@]}"; do
    if [[ ${#find_pattern[@]} -gt 0 ]]; then
        find_pattern+=(-o)          # 除第一个条件外，其余前面加 -o
    fi
    find_pattern+=(-iname "*.$ext")
done

# 构造最终 find 参数（全局选项 -maxdepth 必须放在其他测试之前）
find_args=("$src_dir")
if [[ "$recursive" != true ]]; then
    find_args+=(-maxdepth 1)
fi
find_args+=(-type f '(' "${find_pattern[@]}" ')')
# ==================================================

echo "开始缩放: 源目录='$src_dir' 目标目录='$dest_dir' 缩放比例=${percentage}%"
echo "使用 parallel 并行处理..."

# 使用 find 输出文件列表（以 null 分隔），parallel 接收并执行缩放命令
find "${find_args[@]}" -print0 2>/dev/null | \
    parallel --progress -0 --will-cite convert {} -resize "${percentage}%" "${dest_dir}/{/}"

# 检查任务是否全部完成（parallel 默认会报告错误）
if [[ $? -eq 0 ]]; then
    echo "完成！缩放后的图片已保存到 '$dest_dir'"
else
    echo "处理过程中出现错误，请检查输出。"
    exit 1
fi
