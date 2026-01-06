#!/bin/bash

# 检查是否提供了目录参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <manifests目录>"
    exit 1
fi

MANIFEST_DIR="$1"

# 检查目录是否存在
if [ ! -d "$MANIFEST_DIR" ]; then
    echo "错误: 目录 '$MANIFEST_DIR' 不存在。"
    exit 1
fi

echo "Start testing..."

# kubeconform - 检查 K8s API 规范
echo "Start kubeconform (K8s规范检查)"
kubeconform --verbose $(find "$MANIFEST_DIR" -maxdepth 1 -type f -name "*.yaml" | sort -V) > kubeconform.log 2>&1
echo "kubeconform 结果已保存到  kubeconform.log"

# datree - 安全检查
echo "Start datree (安全检查)"
datree test $(find "$MANIFEST_DIR" -maxdepth 1 -type f -name "*.yaml" | sort -V) > datree.log 2>&1
echo "datree结果已保存到 datree.log"

echo "测试完成！"

chmod 666 kubeconform.log
chmod 666 datree.log

