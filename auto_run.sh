#!/bin/bash

# 检查是否提供了目录参数
if [ $# -ne 1 ]; then
	echo "Error, you must and only use 1 argument"
    echo "用法: $0 <manifests目录>"
    exit 1
fi

MANIFEST_DIR="$1"
DIR_NAME="${MANIFEST_DIR##*/}"

mkdir -p "./logs"  # 自动创建logs目录
LOG_FILE="./logs/$DIR_NAME-k8s-dry-run.log"

# 清空或创建日志文件
> "$LOG_FILE"

echo "开始执行 K8s 资源配置文件校验" | tee -a "$LOG_FILE"
echo "检查目录: $MANIFEST_DIR" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# 查找所有 YAML 文件
yaml_files=$(find "$MANIFEST_DIR" -type f \( -name "*.yaml" -o -name "*.yml" \))

if [ -z "$yaml_files" ]; then
    echo "未找到 YAML 文件" | tee -a "$LOG_FILE"
    exit 0
fi

echo "找到 $(echo "$yaml_files" | wc -l) 个 YAML 文件" | tee -a "$LOG_FILE"

# 计数器
success_count=0
fail_count=0

# 遍历每个文件
for file in $yaml_files; do
    filename=$(basename "$file")
    
    echo "" | tee -a "$LOG_FILE"
    echo "开始处理文件: $filename" | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────" | tee -a "$LOG_FILE"
    
    # 执行 dry-run，将原始输出同时显示并记录到日志
    # 使用 2>&1 将 stderr 重定向到 stdout，确保捕获所有输出
    echo "执行命令: kubectl apply --dry-run=client -f \"$file\"" | tee -a "$LOG_FILE"
    echo "------------------------" | tee -a "$LOG_FILE"
    
    # 执行命令并捕获所有输出和退出状态
    kubectl apply --dry-run=client -f "$file" 2>&1 | tee -a "$LOG_FILE"
    
    # 获取 kubectl 命令的实际退出状态
    k8s_exit_code=${PIPESTATUS[0]}
    
    echo "------------------------" | tee -a "$LOG_FILE"
    
    if [ $k8s_exit_code -eq 0 ]; then
        echo "✅ $filename 校验成功" | tee -a "$LOG_FILE"
        ((success_count++))
    else
        echo "❌ $filename 校验失败" | tee -a "$LOG_FILE"
        ((fail_count++))
    fi
done

echo ""
echo "=========================================="
echo "校验完成!"
echo "成功: $success_count 个文件"
echo "失败: $fail_count 个文件"
echo "详细日志: $LOG_FILE"
