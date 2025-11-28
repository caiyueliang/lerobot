#!/bin/bash

# train.sh - 参数转换和过滤脚本

# 初始化参数数组
final_args=()

# # 第一个参数是模型名称
# if [[ $# -gt 0 ]]; then
#     model_name="$1"
#     shift
# else
#     echo "错误: 需要指定模型名称"
#     exit 1
# fi

# 遍历所有传入的参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_path=*|--data-path=*)
            # 将 --data_path 或 --data-path 映射为 --data.repo-id
            value="${1#*=}"
            final_args+=(--dataset.repo_id="$value")
            shift
            ;;
        --output_path=*|--output-path=*)
            # 将 --output_path 或 --output-path 映射为 --checkpoint-base-dir
            value="${1#*=}"
            final_args+=(--output_dir="$value")
            output_dir="$value"  # 记录 output 目录
            shift
            ;;
        --val_split_ratio=*|--val-split-ratio=*)
            # 丢弃这些参数
            shift
            ;;
        --continuing_training=*|--continuing-training=*)
            # 丢弃这些参数
            shift
            ;;
        --data_path|--data-path)
            # 处理 --data_path value 格式（带空格）
            if [[ $# -gt 1 ]]; then
                final_args+=(--dataset.repo_id="$2")
                shift 2
            else
                echo "错误: --data_path 需要参数值"
                exit 1
            fi
            ;;
        --output_path|--output-path)
            # 处理 --output_path value 格式（带空格）
            if [[ $# -gt 1 ]]; then
                final_args+=(--output_dir="$2")
                output_dir="$2"  # 记录 output 目录
                shift 2
            else
                echo "错误: --output_path 需要参数值"
                exit 1
            fi
            ;;
        --val_split_ratio|--val-split-ratio|--continuing_training|--continuing-training)
            # 丢弃这些参数（带空格格式）
            if [[ $# -gt 1 ]]; then
                shift 2
            else
                shift
            fi
            ;;
        *)
            # 其他参数直接透传
            final_args+=("$1")
            shift
            ;;
    esac
done

# ===== 如果 output_dir 被设置且目录存在，则删除它，否则lerobot会报错 =====
if [[ -n "$output_dir" && -e "$output_dir" ]]; then
    echo "【警告】输出目录已存在，正在删除: $output_dir"
    rm -rf "$output_dir"
    sleep 1
    if [[ $? -ne 0 ]]; then
        echo "错误: 无法删除输出目录 $output_dir"
        exit 1
    fi
fi

# ================================================================
# 打印转换后的命令（用于调试）
echo "【转换后的命令】"
echo "python src/lerobot/scripts/lerobot_train.py ${final_args[@]}"
echo ""

# 执行训练命令
python src/lerobot/scripts/lerobot_train.py "${final_args[@]}"