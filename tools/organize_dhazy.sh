#!/bin/bash
set -euo pipefail

DATA_DIR="${1:-.}"
TRAIN_DIR="${2:-./train}"
HAZY_DIR="$TRAIN_DIR/hazy"
CLEAR_DIR="$TRAIN_DIR/clear"
DEPTH_DIR="$TRAIN_DIR/depth"

TMP_GT="/tmp/dhazy_gt"
TMP_HAZY="/tmp/dhazy_hazy"

command -v unzip >/dev/null 2>&1 || { echo "错误: 未找到 unzip"; exit 1; }
command -v convert &>/dev/null 2>&1 || { echo "错误: 未找到 convert (ImageMagick)"; exit 1; }
command -v parallel &>/dev/null 2>&1 || { echo "错误: 未找到 parallel (GNU parallel)"; exit 1; }

# 清理旧临时文件
rm -rf "$TMP_GT" "$TMP_HAZY"
mkdir -p "$HAZY_DIR" "$CLEAR_DIR" "$DEPTH_DIR"

echo "解压zip文件..."

# 并行解压：同时启动两个 unzip 进程
unzip -q "$DATA_DIR/NYU_GT.zip" -d "$TMP_GT" &
pid_gt=$!
unzip -q "$DATA_DIR/NYU_Hazy.zip" -d "$TMP_HAZY" &
pid_hazy=$!

# 等待两个解压完成，并检查返回码
wait $pid_gt
gt_exit=$?
wait $pid_hazy
hazy_exit=$?

# 检查解压是否成功
if [ $gt_exit -ne 0 ] || [ $hazy_exit -ne 0 ]; then
    echo "错误：解压失败 (GT=${gt_exit}, Hazy=${hazy_exit})" >&2
    exit 1
fi
echo "解压完成。"

# ---------- 提取场景 ID ----------
cd "$TMP_GT"
scene_ids=($(find . -maxdepth 1 -name '*_Image_.bmp' -exec basename {} \; | sed 's/^\([0-9]*\)_.*/\1/' | sort -n -u))
cd - > /dev/null
echo "共找到 ${#scene_ids[@]} 个场景"

# ---------- 并行转换 ----------
work() {
    local idx=$1 id=$2
    local new_name=$(printf "%04d.png" "$idx")
    local clear_src="$TMP_GT/${id}_Image_.bmp"
    local depth_src="$TMP_GT/${id}_Depth_.bmp"
    local hazy_src="$TMP_HAZY/${id}_Hazy.bmp"

    [[ -f "$clear_src" ]] && convert "$clear_src" "$CLEAR_DIR/$new_name" || echo "警告：缺少 $clear_src" >&2
    [[ -f "$depth_src" ]] && convert "$depth_src" "$DEPTH_DIR/$new_name" || echo "警告：缺少 $depth_src" >&2
    [[ -f "$hazy_src"  ]] && convert "$hazy_src"  "$HAZY_DIR/$new_name"  || echo "警告：缺少 $hazy_src" >&2
}
export -f work
export TMP_GT TMP_HAZY CLEAR_DIR HAZY_DIR DEPTH_DIR

parallel -j$(nproc) --eta --xapply work {1} {2} ::: "${!scene_ids[@]}" ::: "${scene_ids[@]}"

# ---------- 清理 ----------
rm -rf "$TMP_GT" "$TMP_HAZY"
echo "完成！训练数据已保存到 $TRAIN_DIR"
