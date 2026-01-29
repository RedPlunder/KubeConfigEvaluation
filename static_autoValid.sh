#!/bin/bash

# 检查是否提供了目录参数
if [ $# -ne 1 ]; then
	echo "Error, you must and only use 1 argument"
    echo "用法: $0 <manifests目录>"
    exit 1
fi

MANIFEST_DIR="$1"
DIR_NAME="${MANIFEST_DIR##*/}"

# 检查目录是否存在
if [ ! -d "$MANIFEST_DIR" ]; then
    echo "错误: 目录 '$MANIFEST_DIR' 不存在。"
    exit 1
fi

mkdir -p "./logs"  # 自动创建logs目录

echo "Start testing..."

# kubeconform - 检查 K8s API 规范
echo "Start kubeconform (K8s规范检查)"
kubeconform --verbose $(find "$MANIFEST_DIR" -maxdepth 1 -type f -name "*.yaml" | sort -V) > ./logs/$DIR_NAME-kubeconform.log 2>&1
echo "kubeconform 结果已保存到  ./logs/$DIR_NAME-kubeconform.log"

# datree - 安全检查
echo "Start datree (安全检查)"
datree test $(find "$MANIFEST_DIR" -maxdepth 1 -type f -name "*.yaml" | sort -V) > ./logs/$DIR_NAME-datree.log 2>&1
# 保留从第一个匹配 ">>  File:" 的行开始的所有内容
# Fliter messy code.
sed -n '/>>  File:/,$p' "./logs/$DIR_NAME-datree.log" > "./logs/$DIR_NAME-datree.log.tmp"
mv "./logs/$DIR_NAME-datree.log.tmp" "./logs/$DIR_NAME-datree.log"
echo "datree结果已保存到 ./logs/$DIR_NAME-datree.log"

echo "测试完成！"

chmod 666 ./logs/$DIR_NAME-kubeconform.log
chmod 666 ./logs/$DIR_NAME-datree.log
