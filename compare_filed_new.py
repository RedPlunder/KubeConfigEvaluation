#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import argparse
import re
import os
import glob
from typing import Dict, Any, Set, List, Optional

try:
    import yaml  # PyYAML
    import pandas as pd
except ImportError:
    print("Please `pip install pyyaml pandas` first.", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    from scipy.stats import ttest_rel, shapiro, wilcoxon, binomtest
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

def preprocess_helm_template(yaml_content):
    """简单预处理 Helm 模板内容，防止 yaml_load() """
    # 移除 {{- ... }} 和 {{ ... }} 块
    content = re.sub(r'\{\{-\s*.+?\s*\}\}', '', yaml_content, flags=re.DOTALL)
    content = re.sub(r'\{\{\s*.+?\s*\}\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\.\.\.', '', content, flags=re.DOTALL)

    # 清理可能产生的空行
    lines = content.split('\n')
    cleaned_lines = [line for line in lines if line.strip() != '']
    return '\n'.join(cleaned_lines)

def get_all_keys(data: Any, parent_key: str = '', separator: str = '.') -> Set[str]:
    """
    递归获取 YAML 数据中的所有字段键名（统一转换为小写以解决大小写不匹配问题）
    """
    keys = set()
    
    if isinstance(data, dict):
        for key, value in data.items():
            # 将键名转换为小写
            key_lower = key.lower()
            new_key = f"{parent_key}{separator}{key_lower}" if parent_key else key_lower
            
            # 添加当前键
            keys.add(new_key)
            
            # 递归处理嵌套字典
            if isinstance(value, (dict, list)):
                keys.update(get_all_keys(value, new_key, separator))
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_key = f"{parent_key}{separator}[{i}]" if parent_key else f"[{i}]"
            
            # 递归处理列表中的元素
            if isinstance(item, (dict, list)):
                keys.update(get_all_keys(item, new_key, separator))
    
    return keys

def compute_gold_keys_from_yaml(gold_yaml: Optional[str]) -> List[str]:
    """Parse a gold YAML snippet and return all flattened keys (lowercase)."""
    if not gold_yaml or not gold_yaml.strip():
        return []
    try:
        parsed = yaml.safe_load(preprocess_helm_template(gold_yaml))
        if parsed is None:
            return []
        return list(get_all_keys(parsed))
    except Exception:
        return []

def build_zero_success_payload(record_index: int,
                               gold_keys: List[str],
                               note: str,
                               extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a zero-coverage success payload with a note describing the reason."""
    total_fields = len(gold_keys)
    payload = {
        'record_index': record_index,
        'total_fields_in_gold': total_fields,
        'existing_fields_in_pred': 0,
        'coverage_percentage': 0.0,
        'missing_fields_count': total_fields,
        'missing_fields': str(gold_keys) if gold_keys else '',
        'note': note
    }
    if extra:
        payload.update(extra)
    return payload

def compare_yaml_fields_single(yaml_str_a: str, yaml_str_b: str) -> Dict[str, Any]:
    """
    比较两个YAML字符串的字段存在性(单次比较)
    
    Args:
        yaml_str_a: 第一个YAML字符串 (gold/reference_answer)
        yaml_str_b: 第二个YAML字符串 (pred/generated_response)
    
    Returns:
        包含比较结果的字典
    """
    # 处理空值或非字符串输入
    if not isinstance(yaml_str_a, str) or not yaml_str_a.strip():
        return {'error': 'gold内容为空或无效'}
    if not isinstance(yaml_str_b, str) or not yaml_str_b.strip():
        return {'error': 'pred内容为空或无效'}

    yaml_str_a = preprocess_helm_template(yaml_str_a)
    yaml_str_b = preprocess_helm_template(yaml_str_b)

    try:
        # 解析YAML字符串
        data_a = yaml.safe_load(yaml_str_a)
        data_b = yaml.safe_load(yaml_str_b)
        
        # 检查解析结果是否为None
        if data_a is None:
            return {'error': 'gold内容解析后为空'}
        if data_b is None:
            return {'error': 'pred内容解析后为空'}
        
        # 获取所有字段键名
        keys_a = get_all_keys(data_a)
        keys_b = get_all_keys(data_b)
        
        # 找出在A中存在但在B中不存在的字段
        missing_in_b = keys_a - keys_b
        common_fields = keys_a & keys_b
        
        # 计算比例
        total_fields_a = len(keys_a)
        existing_fields_count = len(common_fields)
        coverage_ratio = existing_fields_count / total_fields_a if total_fields_a > 0 else 0
        
        # 返回比较结果
        return {
            'total_fields_in_a': total_fields_a,
            'existing_fields_in_b': existing_fields_count,
            'missing_fields_in_b': list(missing_in_b),
            'coverage_ratio': coverage_ratio,
            'coverage_percentage': round(coverage_ratio * 100, 2),
            'all_fields_in_a': list(keys_a),
            'common_fields': list(common_fields),
            'error': None
        }
        
    except yaml.YAMLError as e:
        return {'error': f'YAML解析错误: {str(e)}'}
    except Exception as e:
        return {'error': f'比较过程中发生错误: {str(e)}'}

# Gold 用普通 yaml 格式
YAML_BLOCK_REGEX_GOLD = r"```yaml\n(.*?)```"
# # Pred 匹配：yaml:complete / yaml: complete / 普通yaml
# YAML_BLOCK_REGEX_PRED = r"```yaml(?::[ ]?complete)?\n(.*?)```"
# Pred 只匹配：yaml:complete / yaml: complete
YAML_BLOCK_REGEX_PRED = r"```yaml:[ ]?complete\n(.*?)```"

def extract_yaml(text: str, is_pred: bool = False) -> Optional[List[str]]:
    """
    从文本中提取所有yaml代码块，并分割多文档YAML（用---分隔）
    
    Args:
        text: 要提取的文本
        is_pred: 如果为True，使用 yaml: complete 正则；否则使用普通 yaml 正则
    """
    if not text or pd.isna(text):
        return None
    
    text = str(text)
    # 根据类型选择正则
    regex = YAML_BLOCK_REGEX_PRED if is_pred else YAML_BLOCK_REGEX_GOLD
    # 提取所有yaml代码块
    matches = re.findall(regex, text, re.DOTALL)
    if not matches:
        return None
    
    # 处理每个代码块，分割多文档YAML
    all_yamls = []
    for yaml_block in matches:
        yaml_block = yaml_block.strip()
        if not yaml_block:
            continue
        
        # 按 --- 分割多文档YAML（支持多种格式：\n---\n, \n---, ---\n）
        # 使用正则表达式匹配 --- 分隔符（前后可能有空白）
        docs = re.split(r'\n\s*---\s*\n', yaml_block)
        
        for doc in docs:
            doc = doc.strip()
            # 移除文档开头的 ---（如果存在）
            if doc.startswith('---'):
                # 检查是单独的 --- 行还是 --- 后面有内容
                lines = doc.split('\n', 1)
                if len(lines) > 1 and lines[0].strip() == '---':
                    doc = lines[1].strip()
                elif lines[0].strip() == '---':
                    continue  # 跳过只有 --- 的文档
            
            if doc:  # 只添加非空文档
                all_yamls.append(doc)
    
    return all_yamls if all_yamls else None

def extract_yaml_direct(text: str) -> Optional[List[str]]:
    """
    直接将文本作为YAML处理，支持多文档分割
    """
    if not text or pd.isna(text):
        return None
    
    text = str(text).strip()
    if not text:
        return None
    
    all_yamls = []
    # 按 --- 分割多文档YAML
    docs = re.split(r'\n\s*---\s*\n', text)
    
    for doc in docs:
        doc = doc.strip()
        if doc.startswith('---'):
            lines = doc.split('\n', 1)
            if len(lines) > 1 and lines[0].strip() == '---':
                doc = lines[1].strip()
            elif lines[0].strip() == '---':
                continue
        
        if doc:
            all_yamls.append(doc)
            
    return all_yamls if all_yamls else None

def load_yaml_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------
# Mode 1: CSV + JSON folder
# ---------------------------

def process_mode1(csv_path: str, json_folder: str, gold_column: str = 'newAnswer Body', output_file: Optional[str] = None, limit: Optional[int] = None):
    """
    模式1：从CSV提取gold，从文件夹中的JSON文件提取pred
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        total_records = len(df)
        print(f"成功读取CSV文件, 共 {total_records} 条记录")
        
        # 如果指定了limit，只处理前limit条
        if limit is not None and limit > 0:
            df = df.head(limit)
            print(f"限制处理前 {limit} 条记录")
    except Exception as e:
        print(f"读取CSV文件失败: {e}", file=sys.stderr)
        return pd.DataFrame()
    
    # 检查列是否存在
    if gold_column not in df.columns:
        print(f"错误: CSV文件中不存在列 '{gold_column}'", file=sys.stderr)
        return pd.DataFrame()
    
    # 存储比较结果
    results = []
    
    # 处理每一行（按索引配对）
    for idx, row in df.iterrows():
        # 从CSV提取gold（提取所有yaml代码块）
        gold_text = row.get(gold_column, '')
        gold_yamls = extract_yaml(gold_text)
        
        if not gold_yamls:
            results.append({
                'record_index': idx,
                'success': None,
                'failure': 'No yaml found in gold'
            })
            continue
        
        # 验证 gold (reference_answer) 是否是有效的 YAML
        has_valid_gold = False
        valid_gold_yaml = None
        for gold_yaml in gold_yamls:
            sanity_check = compare_yaml_fields_single(gold_yaml, gold_yaml)
            if not sanity_check.get('error'):
                has_valid_gold = True
                valid_gold_yaml = gold_yaml
                break
        
        if not has_valid_gold:
            results.append({
                'record_index': idx,
                'success': None,
                'failure': 'Reference answer is not a valid YAML'
            })
            continue

        gold_keys = compute_gold_keys_from_yaml(valid_gold_yaml)
        
        # 找到对应的JSON文件（idx.json）
        json_file = os.path.join(json_folder, f"{idx}.json")
        
        if not os.path.exists(json_file):
            zero_payload = build_zero_success_payload(
                idx, gold_keys, 'pred_json_missing',
                extra={'json_file': f'{idx}.json'}
            )
            results.append({
                'record_index': idx,
                'success': json.dumps(zero_payload, ensure_ascii=False),
                'failure': None
            })
            continue
        
        # 从对应的JSON文件提取pred
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 处理JSON数据：可能是列表或字典
            if isinstance(json_data, list):
                # 如果是列表，取第一个元素（或遍历所有元素）
                if len(json_data) > 0:
                    json_data = json_data[0]
                else:
                    zero_payload = build_zero_success_payload(
                        idx, gold_keys, 'pred_json_empty_list',
                        extra={'json_file': os.path.basename(json_file)}
                    )
                    results.append({
                        'record_index': idx,
                        'success': json.dumps(zero_payload, ensure_ascii=False),
                        'failure': None
                    })
                    continue
            
            # 提取output字段中的yaml
            if not isinstance(json_data, dict):
                zero_payload = build_zero_success_payload(
                    idx,
                    gold_keys,
                    f'pred_json_not_dict({type(json_data).__name__})',
                    extra={'json_file': os.path.basename(json_file)}
                )
                results.append({
                    'record_index': idx,
                    'success': json.dumps(zero_payload, ensure_ascii=False),
                    'failure': None
                })
                continue
            
            output_text = json_data.get('output', '')
            pred_yamls = extract_yaml(output_text)
            
            if not pred_yamls:
                zero_payload = build_zero_success_payload(
                    idx,
                    gold_keys,
                    'pred_yaml_missing',
                    extra={'json_file': os.path.basename(json_file)}
                )
                results.append({
                    'record_index': idx,
                    'success': json.dumps(zero_payload, ensure_ascii=False),
                    'failure': None
                })
                continue
            
            # 比较所有可能的组合，选择最大的coverage
            best_coverage = 0
            best_result = None
            has_valid_pred = False
            error_msg = None
            
            for gold_yaml in gold_yamls:
                for pred_yaml in pred_yamls:
                    comparison_result = compare_yaml_fields_single(gold_yaml, pred_yaml)
                    
                    if comparison_result.get('error'):
                        if error_msg is None:
                            error_msg = comparison_result.get('error')
                        continue
                    
                    has_valid_pred = True
                    coverage = comparison_result.get('coverage_percentage', 0)
                    if coverage >= best_coverage:
                        best_coverage = coverage
                        best_result = {
                            'record_index': idx,
                            'json_file': os.path.basename(json_file),
                            'total_fields_in_gold': comparison_result.get('total_fields_in_a', 0),
                            'existing_fields_in_pred': comparison_result.get('existing_fields_in_b', 0),
                            'coverage_percentage': coverage,
                            'missing_fields_count': len(comparison_result.get('missing_fields_in_b', [])),
                            'missing_fields': str(comparison_result.get('missing_fields_in_b', [])) if comparison_result.get('missing_fields_in_b') else ''
                        }
            
            if not has_valid_pred:
                first_valid_gold = None
                for gold_yaml in gold_yamls:
                    sanity_check = compare_yaml_fields_single(gold_yaml, gold_yaml)
                    if not sanity_check.get('error'):
                        first_valid_gold = gold_yaml
                        break
                
                if first_valid_gold:
                    gold_keys = list(get_all_keys(yaml.safe_load(preprocess_helm_template(first_valid_gold))))
                    total_fields = len(gold_keys)
                    
                    best_result = {
                        'record_index': idx,
                        'json_file': os.path.basename(json_file),
                        'total_fields_in_gold': total_fields,
                        'existing_fields_in_pred': 0,
                        'coverage_percentage': 0.0,
                        'missing_fields_count': total_fields,
                        'missing_fields': str(gold_keys) if gold_keys else '',
                        'note': 'pred_comparison_error'
                    }
            
            if best_result:
                results.append({
                    'record_index': idx,
                    'success': json.dumps(best_result, ensure_ascii=False),
                    'failure': None
                })
            else:
                results.append({
                    'record_index': idx,
                    'success': None,
                    'failure': error_msg if error_msg else 'No valid comparison result'
                })
            
        except Exception as e:
            print(f"处理JSON文件 {json_file} 时出错: {e}", file=sys.stderr)
            zero_payload = build_zero_success_payload(
                idx,
                gold_keys,
                'pred_json_error',
                extra={'json_file': os.path.basename(json_file)}
            )
            results.append({
                'record_index': idx,
                'success': json.dumps(zero_payload, ensure_ascii=False),
                'failure': None
            })
            continue
        
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{len(df)} 条记录")
    
    results_df = pd.DataFrame(results)
    print_statistics(results_df, len(df))
    
    if output_file:
        try:
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\n结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存结果文件失败: {e}", file=sys.stderr)
    
    return results_df

# ---------------------------
# Mode 2: CSV only
# ---------------------------

def process_mode2(csv_path: str, 
                  gold_column: str = 'gpt_Generated_Response',
                  pred_column: str = 'gpt_Refined_Response',
                  output_file: Optional[str] = None,
                  limit: Optional[int] = None,
                  model_name: Optional[str] = None):
    """
    模式2：从CSV中提取gold和pred
    如果指定 model_name，会创建对应文件夹并保存每条记录中 coverage 最大的 pred YAML
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        total_records = len(df)
        print(f"成功读取CSV文件, 共 {total_records} 条记录")
        
        if limit is not None and limit > 0:
            df = df.head(limit)
            print(f"限制处理前 {limit} 条记录")
    except Exception as e:
        print(f"读取CSV文件失败: {e}", file=sys.stderr)
        return pd.DataFrame()
    
    # 检查列是否存在
    if gold_column not in df.columns:
        print(f"错误: CSV文件中不存在列 '{gold_column}'", file=sys.stderr)
        return pd.DataFrame()
    if pred_column not in df.columns:
        print(f"错误: CSV文件中不存在列 '{pred_column}'", file=sys.stderr)
        return pd.DataFrame()
    
    # 如果指定了 model_name，创建输出文件夹
    yaml_output_dir = None
    if model_name:
        yaml_output_dir = os.path.join(os.path.dirname(csv_path) or '.', model_name)
        os.makedirs(yaml_output_dir, exist_ok=True)
        print(f"将保存最佳 pred YAML 到: {yaml_output_dir}/")
    
    results = []
    
    for idx, row in df.iterrows():
        gold_text = row.get(gold_column, '')
        pred_text = row.get(pred_column, '')
        
        gold_yamls = extract_yaml(gold_text, is_pred=False)  # Gold 用普通 yaml
        pred_yamls = extract_yaml(pred_text, is_pred=True)   # Pred 用 yaml: complete
        
        if not gold_yamls:
            results.append({
                'record_index': idx,
                'success': None,
                'failure': 'No yaml found in gold'
            })
            continue
        
        has_valid_gold = False
        valid_gold_yaml = None
        for gold_yaml in gold_yamls:
            sanity_check = compare_yaml_fields_single(gold_yaml, gold_yaml)
            if not sanity_check.get('error'):
                has_valid_gold = True
                valid_gold_yaml = gold_yaml
                break
        
        if not has_valid_gold:
            results.append({
                'record_index': idx,
                'success': None,
                'failure': 'Reference answer is not a valid YAML'
            })
            continue
        
        gold_keys = compute_gold_keys_from_yaml(valid_gold_yaml)
        
        if not pred_yamls:
            zero_payload = build_zero_success_payload(
                idx,
                gold_keys,
                'pred_yaml_missing'
            )
            results.append({
                'record_index': idx,
                'success': json.dumps(zero_payload, ensure_ascii=False),
                'failure': None
            })
            continue
        
        best_coverage = 0
        best_result = None
        best_pred_yaml = None  # 记录 coverage 最大的 pred YAML
        has_valid_pred = False
        error_msg = None
        
        for subyaml_a in gold_yamls:
            for subyaml_b in pred_yamls:
                comparison_result = compare_yaml_fields_single(subyaml_a, subyaml_b)
                
                if comparison_result.get('error'):
                    if error_msg is None:
                        error_msg = comparison_result.get('error')
                    continue
                
                has_valid_pred = True
                coverage = comparison_result.get('coverage_percentage', 0)
                if coverage >= best_coverage:
                    best_coverage = coverage
                    best_pred_yaml = subyaml_b  # 记录最佳 pred YAML
                    best_result = {
                        'record_index': idx,
                        'total_fields_in_gold': comparison_result.get('total_fields_in_a', 0),
                        'existing_fields_in_pred': comparison_result.get('existing_fields_in_b', 0),
                        'coverage_percentage': coverage,
                        'missing_fields_count': len(comparison_result.get('missing_fields_in_b', [])),
                        'missing_fields': str(comparison_result.get('missing_fields_in_b', [])) if comparison_result.get('missing_fields_in_b') else ''
                    }
        
        # 保存最佳 pred YAML 到文件
        if yaml_output_dir and best_pred_yaml:
            yaml_file_path = os.path.join(yaml_output_dir, f"{idx + 1}.yaml")
            try:
                with open(yaml_file_path, 'w', encoding='utf-8') as f:
                    f.write(best_pred_yaml)
            except Exception as e:
                print(f"保存 {yaml_file_path} 失败: {e}", file=sys.stderr)
        
        if not has_valid_pred:
            first_valid_gold = None
            for gold_yaml in gold_yamls:
                sanity_check = compare_yaml_fields_single(gold_yaml, gold_yaml)
                if not sanity_check.get('error'):
                    first_valid_gold = gold_yaml
                    break
            
            if first_valid_gold:
                gold_keys_local = list(get_all_keys(yaml.safe_load(preprocess_helm_template(first_valid_gold))))
                total_fields = len(gold_keys_local)
                
                best_result = {
                    'record_index': idx,
                    'total_fields_in_gold': total_fields,
                    'existing_fields_in_pred': 0,
                    'coverage_percentage': 0.0,
                    'missing_fields_count': total_fields,
                    'missing_fields': str(gold_keys_local) if gold_keys_local else '',
                    'note': 'pred_comparison_error'
                }
        
        if best_result:
            results.append({
                'record_index': idx,
                'success': json.dumps(best_result, ensure_ascii=False),
                'failure': None
            })
        else:
            results.append({
                'record_index': idx,
                'success': None,
                'failure': error_msg if error_msg else 'No valid comparison result'
            })
        
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{len(df)} 条记录")
    
    results_df = pd.DataFrame(results)
    print_statistics(results_df, len(df))
    
    # 打印 YAML 保存统计
    if yaml_output_dir:
        saved_count = len([f for f in os.listdir(yaml_output_dir) if f.endswith('.yaml')])
        print(f"\n已保存 {saved_count} 个最佳 pred YAML 到: {yaml_output_dir}/")
    
    if output_file:
        try:
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\n结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存结果文件失败: {e}", file=sys.stderr)
    
    return results_df

def process_mode_direct(csv_path: str,
                        gold_column: str,
                        pred_column: str,
                        output_file: Optional[str] = None,
                        limit: Optional[int] = None):
    """
    模式3（重构后）：单CSV，两个字段包含直接的YAML内容（不提取```yaml）
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        total_records = len(df)
        print(f"成功读取CSV文件, 共 {total_records} 条记录")
        
        if limit is not None and limit > 0:
            df = df.head(limit)
            print(f"限制处理前 {limit} 条记录")
    except Exception as e:
        print(f"读取CSV文件失败: {e}", file=sys.stderr)
        return pd.DataFrame()
    
    # 检查列是否存在
    if gold_column not in df.columns:
        print(f"错误: CSV文件中不存在列 '{gold_column}'", file=sys.stderr)
        return pd.DataFrame()
    if pred_column not in df.columns:
        print(f"错误: CSV文件中不存在列 '{pred_column}'", file=sys.stderr)
        return pd.DataFrame()
    
    results = []
    
    for idx, row in df.iterrows():
        gold_text = row.get(gold_column, '')
        pred_text = row.get(pred_column, '')
        
        # 使用extract_yaml_direct而不是extract_yaml
        gold_yamls = extract_yaml_direct(gold_text)
        pred_yamls = extract_yaml_direct(pred_text)
        
        if not gold_yamls:
            results.append({
                'record_index': idx,
                'success': None,
                'failure': 'No yaml found in gold'
            })
            continue
        
        if not pred_yamls:
            gold_keys = []
            total_fields = 0
            if valid_gold_yaml:
                try:
                    gold_keys = get_all_keys(yaml.safe_load(preprocess_helm_template(valid_gold_yaml)))
                except Exception:
                    gold_keys = []
                total_fields = len(gold_keys)
            zero_result = {
                'record_index': idx,
                'total_fields_in_gold': total_fields,
                'existing_fields_in_pred': 0,
                'coverage_percentage': 0.0,
                'missing_fields_count': total_fields,
                'missing_fields': str(list(gold_keys)) if gold_keys else '',
                'note': 'pred_yaml_missing'
            }
            results.append({
                'record_index': idx,
                'success': json.dumps(zero_result, ensure_ascii=False),
                'failure': None
            })
            continue
        
        has_valid_gold = False
        valid_gold_yaml = None
        for gold_yaml in gold_yamls:
            sanity_check = compare_yaml_fields_single(gold_yaml, gold_yaml)
            if not sanity_check.get('error'):
                has_valid_gold = True
                valid_gold_yaml = gold_yaml
                break
        
        if not has_valid_gold:
            results.append({
                'record_index': idx,
                'success': None,
                'failure': 'Reference answer is not a valid YAML'
            })
            continue
        
        best_coverage = 0
        best_result = None
        has_valid_pred = False
        error_msg = None
        
        for subyaml_a in gold_yamls:
            for subyaml_b in pred_yamls:
                comparison_result = compare_yaml_fields_single(subyaml_a, subyaml_b)
                
                if comparison_result.get('error'):
                    if error_msg is None:
                        error_msg = comparison_result.get('error')
                    continue
                
                has_valid_pred = True
                coverage = comparison_result.get('coverage_percentage', 0)
                if coverage >= best_coverage:
                    best_coverage = coverage
                    best_result = {
                        'record_index': idx,
                        'total_fields_in_gold': comparison_result.get('total_fields_in_a', 0),
                        'existing_fields_in_pred': comparison_result.get('existing_fields_in_b', 0),
                        'coverage_percentage': coverage,
                        'missing_fields_count': len(comparison_result.get('missing_fields_in_b', [])),
                        'missing_fields': str(comparison_result.get('missing_fields_in_b', [])) if comparison_result.get('missing_fields_in_b') else ''
                    }
        
        if not has_valid_pred:
            first_valid_gold = None
            for gold_yaml in gold_yamls:
                sanity_check = compare_yaml_fields_single(gold_yaml, gold_yaml)
                if not sanity_check.get('error'):
                    first_valid_gold = gold_yaml
                    break
            
            if first_valid_gold:
                gold_keys = get_all_keys(yaml.safe_load(preprocess_helm_template(first_valid_gold)))
                total_fields = len(gold_keys)
                
                best_result = {
                    'record_index': idx,
                    'total_fields_in_gold': total_fields,
                    'existing_fields_in_pred': 0,
                    'coverage_percentage': 0.0,
                    'missing_fields_count': total_fields,
                    'missing_fields': str(list(gold_keys)) if gold_keys else ''
                }
        
        if best_result:
            results.append({
                'record_index': idx,
                'success': json.dumps(best_result, ensure_ascii=False),
                'failure': None
            })
        else:
            results.append({
                'record_index': idx,
                'success': None,
                'failure': error_msg if error_msg else 'No valid comparison result'
            })
        
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{len(df)} 条记录")
    
    results_df = pd.DataFrame(results)
    print_statistics(results_df, len(df))
    
    if output_file:
        try:
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\n结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存结果文件失败: {e}", file=sys.stderr)
    
    return results_df

def process_mode4(gold_csv: str,
                  pred_csv: str,
                  gold_column: str,
                  pred_column: str,
                  output_file: Optional[str] = None,
                  limit: Optional[int] = None):
    """
    模式4：双CSV，Gold需提取yaml block，Pred直接是YAML
    """
    # 读取Gold CSV
    try:
        df_gold = pd.read_csv(gold_csv, encoding="utf-8")
        print(f"成功读取Gold CSV文件, 共 {len(df_gold)} 条记录")
    except Exception as e:
        print(f"读取Gold CSV文件失败: {e}", file=sys.stderr)
        return pd.DataFrame()

    # 读取Pred CSV
    try:
        df_pred = pd.read_csv(pred_csv, encoding="utf-8")
        print(f"成功读取Pred CSV文件, 共 {len(df_pred)} 条记录")
    except Exception as e:
        print(f"读取Pred CSV文件失败: {e}", file=sys.stderr)
        return pd.DataFrame()
    
    # 检查列是否存在
    if gold_column not in df_gold.columns:
        print(f"错误: Gold CSV文件中不存在列 '{gold_column}'", file=sys.stderr)
        return pd.DataFrame()
    if pred_column not in df_pred.columns:
        print(f"错误: Pred CSV文件中不存在列 '{pred_column}'", file=sys.stderr)
        return pd.DataFrame()
    
    total_records = min(len(df_gold), len(df_pred))
    if limit is not None and limit > 0:
        total_records = min(total_records, limit)
        print(f"限制处理前 {total_records} 条记录")

    results = []
    
    for idx in range(total_records):
        gold_text = df_gold[gold_column].iloc[idx]
        pred_text = df_pred[pred_column].iloc[idx]
        
        # Gold提取yaml block，Pred直接解析
        gold_yamls = extract_yaml(gold_text)
        pred_yamls = extract_yaml_direct(pred_text)
        
        if not gold_yamls:
            results.append({
                'record_index': idx,
                'success': None,
                'failure': 'No yaml found in gold'
            })
            continue
        
        has_valid_gold = False
        valid_gold_yaml = None
        for gold_yaml in gold_yamls:
            sanity_check = compare_yaml_fields_single(gold_yaml, gold_yaml)
            if not sanity_check.get('error'):
                has_valid_gold = True
                valid_gold_yaml = gold_yaml
                break
        
        if not has_valid_gold:
            results.append({
                'record_index': idx,
                'success': None,
                'failure': 'Reference answer is not a valid YAML'
            })
            continue
        
        if not pred_yamls:
            gold_keys = []
            total_fields = 0
            if valid_gold_yaml:
                try:
                    gold_keys = get_all_keys(yaml.safe_load(preprocess_helm_template(valid_gold_yaml)))
                except Exception:
                    gold_keys = []
                total_fields = len(gold_keys)
            zero_result = {
                'record_index': idx,
                'total_fields_in_gold': total_fields,
                'existing_fields_in_pred': 0,
                'coverage_percentage': 0.0,
                'missing_fields_count': total_fields,
                'missing_fields': str(list(gold_keys)) if gold_keys else '',
                'note': 'pred_yaml_missing'
            }
            results.append({
                'record_index': idx,
                'success': json.dumps(zero_result, ensure_ascii=False),
                'failure': None
            })
            continue
        
        best_coverage = 0
        best_result = None
        has_valid_pred = False
        error_msg = None
        
        for subyaml_a in gold_yamls:
            for subyaml_b in pred_yamls:
                comparison_result = compare_yaml_fields_single(subyaml_a, subyaml_b)
                
                if comparison_result.get('error'):
                    if error_msg is None:
                        error_msg = comparison_result.get('error')
                    continue
                
                has_valid_pred = True
                coverage = comparison_result.get('coverage_percentage', 0)
                if coverage >= best_coverage:
                    best_coverage = coverage
                    best_result = {
                        'record_index': idx,
                        'total_fields_in_gold': comparison_result.get('total_fields_in_a', 0),
                        'existing_fields_in_pred': comparison_result.get('existing_fields_in_b', 0),
                        'coverage_percentage': coverage,
                        'missing_fields_count': len(comparison_result.get('missing_fields_in_b', [])),
                        'missing_fields': str(comparison_result.get('missing_fields_in_b', [])) if comparison_result.get('missing_fields_in_b') else ''
                    }
        
        if not has_valid_pred:
            first_valid_gold = None
            for gold_yaml in gold_yamls:
                sanity_check = compare_yaml_fields_single(gold_yaml, gold_yaml)
                if not sanity_check.get('error'):
                    first_valid_gold = gold_yaml
                    break
            
            if first_valid_gold:
                gold_keys = get_all_keys(yaml.safe_load(preprocess_helm_template(first_valid_gold)))
                total_fields = len(gold_keys)
                
                best_result = {
                    'record_index': idx,
                    'total_fields_in_gold': total_fields,
                    'existing_fields_in_pred': 0,
                    'coverage_percentage': 0.0,
                    'missing_fields_count': total_fields,
                    'missing_fields': str(list(gold_keys)) if gold_keys else ''
                }
        
        if best_result:
            results.append({
                'record_index': idx,
                'success': json.dumps(best_result, ensure_ascii=False),
                'failure': None
            })
        else:
            results.append({
                'record_index': idx,
                'success': None,
                'failure': error_msg if error_msg else 'No valid comparison result'
            })
        
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{total_records} 条记录")
    
    results_df = pd.DataFrame(results)
    print_statistics(results_df, total_records)
    
    if output_file:
        try:
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\n结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存结果文件失败: {e}", file=sys.stderr)
    
    return results_df

def print_statistics(results_df: pd.DataFrame, total_records: int):
    """打印统计信息（只使用success记录）"""
    print("\n" + "="*60)
    print("比较结果统计")
    print("="*60)
    
    success_records = results_df[results_df['success'].notna()]
    
    if len(success_records) > 0:
        coverage_list = []
        for _, row in success_records.iterrows():
            try:
                success_data = json.loads(row['success'])
                coverage_list.append(success_data.get('coverage_percentage', 0))
            except:
                continue
        
        if coverage_list:
            avg_coverage = sum(coverage_list) / len(coverage_list)
            max_coverage = max(coverage_list)
            min_coverage = min(coverage_list)
            
            print(f"成功比较记录数: {len(success_records)}/{total_records}")
            print(f"失败记录数: {total_records - len(success_records)}")
            print(f"平均覆盖率: {avg_coverage:.2f}%")
            print(f"最高覆盖率: {max_coverage:.2f}%")
            print(f"最低覆盖率: {min_coverage:.2f}%")
            
            coverage_ranges = {
                '100%': sum(1 for c in coverage_list if c == 100),
                '90%-100%': sum(1 for c in coverage_list if c >= 90),
                '80%-89%': sum(1 for c in coverage_list if 80 <= c < 90),
                '70%-79%': sum(1 for c in coverage_list if 70 <= c < 80),
                '60%-69%': sum(1 for c in coverage_list if 60 <= c < 70),
                '低于60%': sum(1 for c in coverage_list if c < 60)
            }
            
            print("\n覆盖率分布:")
            for range_name, count in coverage_ranges.items():
                percentage = (count / len(coverage_list)) * 100 if len(coverage_list) > 0 else 0
                print(f"  {range_name}: {count} 条 ({percentage:.1f}%)")
        else:
            print("没有有效的比较结果")
    else:
        print("没有成功的比较结果")

def process_mode_batch_json(csv_path: str, json_folder: str, gold_column: str = 'newAnswer Body', output_file: Optional[str] = None, limit: Optional[int] = None):
    """
    模式5：从CSV提取gold，从文件夹中的多个JSON文件(0.json, 1.json...)提取pred
    JSON文件包含记录列表。
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        print(f"成功读取CSV文件, 共 {len(df)} 条记录")
    except Exception as e:
        print(f"读取CSV文件失败: {e}", file=sys.stderr)
        return pd.DataFrame()
    
    if gold_column not in df.columns:
        print(f"错误: CSV文件中不存在列 '{gold_column}'", file=sys.stderr)
        return pd.DataFrame()

    # 读取并合并JSON文件
    all_preds = []
    json_pattern = os.path.join(json_folder, "*.json")
    json_files = glob.glob(json_pattern)
    
    # 按文件名中的数字排序
    try:
        json_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except ValueError:
        print("警告: 文件名包含非数字，将按字典序排序")
        json_files.sort()
    
    print(f"找到 {len(json_files)} 个JSON文件: {[os.path.basename(f) for f in json_files]}")
    
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        all_preds.append({
                            'output': item.get('output', ''),
                            'source': os.path.basename(jf)
                        })
                else:
                    print(f"警告: {jf} 不是列表格式，跳过")
        except Exception as e:
            print(f"读取JSON {jf} 失败: {e}")

    print(f"共提取到 {len(all_preds)} 条预测记录")
    
    # 限制处理数量
    total_records = min(len(df), len(all_preds))
    if limit is not None and limit > 0:
        total_records = min(total_records, limit)
        print(f"限制处理前 {total_records} 条记录")
        
    results = []
    
    for idx in range(total_records):
        row = df.iloc[idx]
        pred_record = all_preds[idx]
        
        gold_text = row.get(gold_column, '')
        pred_text = pred_record['output']
        source_file = pred_record['source']
        
        gold_yamls = extract_yaml(gold_text)
        pred_yamls = extract_yaml(pred_text)
        
        if not gold_yamls:
            results.append({
                'record_index': idx,
                'source_file': source_file,
                'success': None,
                'failure': 'No yaml found in gold'
            })
            continue
            
        # 验证 gold
        has_valid_gold = False
        valid_gold_yaml = None
        for gold_yaml in gold_yamls:
            sanity_check = compare_yaml_fields_single(gold_yaml, gold_yaml)
            if not sanity_check.get('error'):
                has_valid_gold = True
                valid_gold_yaml = gold_yaml
                break
        
        if not has_valid_gold:
            results.append({
                'record_index': idx,
                'source_file': source_file,
                'success': None,
                'failure': 'Reference answer is not a valid YAML'
            })
            continue
            
        if not pred_yamls:
            gold_keys = []
            total_fields = 0
            if valid_gold_yaml:
                try:
                    gold_keys = get_all_keys(yaml.safe_load(preprocess_helm_template(valid_gold_yaml)))
                except Exception:
                    gold_keys = []
                total_fields = len(gold_keys)
            zero_result = {
                'record_index': idx,
                'source_file': source_file,
                'total_fields_in_gold': total_fields,
                'existing_fields_in_pred': 0,
                'coverage_percentage': 0.0,
                'missing_fields_count': total_fields,
                'missing_fields': str(list(gold_keys)) if gold_keys else '',
                'note': 'pred_yaml_missing'
            }
            results.append({
                'record_index': idx,
                'source_file': source_file,
                'success': json.dumps(zero_result, ensure_ascii=False),
                'failure': None
            })
            continue
            
        # 比较
        best_coverage = 0
        best_result = None
        has_valid_pred = False
        error_msg = None
        
        for gold_yaml in gold_yamls:
            for pred_yaml in pred_yamls:
                comparison_result = compare_yaml_fields_single(gold_yaml, pred_yaml)
                
                if comparison_result.get('error'):
                    if error_msg is None:
                        error_msg = comparison_result.get('error')
                    continue
                
                has_valid_pred = True
                coverage = comparison_result.get('coverage_percentage', 0)
                if coverage >= best_coverage:
                    best_coverage = coverage
                    best_result = {
                        'record_index': idx,
                        'source_file': source_file,
                        'total_fields_in_gold': comparison_result.get('total_fields_in_a', 0),
                        'existing_fields_in_pred': comparison_result.get('existing_fields_in_b', 0),
                        'coverage_percentage': coverage,
                        'missing_fields_count': len(comparison_result.get('missing_fields_in_b', [])),
                        'missing_fields': str(comparison_result.get('missing_fields_in_b', [])) if comparison_result.get('missing_fields_in_b') else ''
                    }
        
        if not has_valid_pred:
            first_valid_gold = None
            for gold_yaml in gold_yamls:
                sanity_check = compare_yaml_fields_single(gold_yaml, gold_yaml)
                if not sanity_check.get('error'):
                    first_valid_gold = gold_yaml
                    break
            
            if first_valid_gold:
                gold_keys = get_all_keys(yaml.safe_load(preprocess_helm_template(first_valid_gold)))
                total_fields = len(gold_keys)
                
                best_result = {
                    'record_index': idx,
                    'source_file': source_file,
                    'total_fields_in_gold': total_fields,
                    'existing_fields_in_pred': 0,
                    'coverage_percentage': 0.0,
                    'missing_fields_count': total_fields,
                    'missing_fields': str(list(gold_keys)) if gold_keys else ''
                }
        
        if best_result:
            results.append({
                'record_index': idx,
                'source_file': source_file,
                'success': json.dumps(best_result, ensure_ascii=False),
                'failure': None
            })
        else:
            results.append({
                'record_index': idx,
                'source_file': source_file,
                'success': None,
                'failure': error_msg if error_msg else 'No valid comparison result'
            })
            
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{total_records} 条记录")

    results_df = pd.DataFrame(results)
    print_statistics(results_df, total_records)
    
    if output_file:
        try:
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\n结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存结果文件失败: {e}", file=sys.stderr)
            
    return results_df

def process_plot_mode(plot_data: List[str], ref_model: str, output_plot: str):
    """
    绘制模型对比箱线图，计算统计显著性
    plot_data: ["ModelA=path/to/a.csv", "ModelB=path/to/b.csv", ...]
    """
    if not HAS_PLOTTING:
        print("错误: 缺少绘图依赖 (matplotlib, scipy, numpy)，无法运行 plot 模式。", file=sys.stderr)
        return

    print(f"开始绘图处理，共 {len(plot_data)} 个模型...")
    
    # 1. 加载所有数据
    all_dfs = []
    model_names = []
    
    for item in plot_data:
        if "=" not in item:
            print(f"警告: 输入格式错误 '{item}'，应为 'ModelName=CSVPath'，跳过")
            continue
            
        model_name, csv_path = item.split("=", 1)
        model_names.append(model_name)
        
        try:
            df = pd.read_csv(csv_path)
            # 提取 coverage_percentage
            records = []
            for _, row in df.iterrows():
                coverage = 0.0
                error_val = 1 # default error
                
                if pd.isna(row.get('failure')) and pd.notna(row.get('success')):
                    try:
                        success_data = json.loads(row['success'])
                        coverage = float(success_data.get('coverage_percentage', 0))
                        error_val = 0
                    except:
                        pass
                
                records.append({
                    'model_name': model_name,
                    'record_index': row.get('record_index'),
                    'coverage_percentage': coverage,
                    'error_message': error_val
                })
            
            all_dfs.append(pd.DataFrame(records))
            print(f"已加载 {model_name}: {len(records)} 条记录")
            
        except Exception as e:
            print(f"读取文件 {csv_path} 失败: {e}")
            return

    if not all_dfs:
        print("没有有效的数据可供绘图")
        return

    # 合并所有数据
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # 2. 准备绘图数据 (使用每个模型的独立有效记录)
    print("正在准备绘图数据 (Independent Valid Records)...")
    
    model_valid_data = {}
    model_means = {}
    
    unique_models = full_df["model_name"].unique()
    
    for m in unique_models:
        # 筛选该模型成功的记录
        valid_rows = full_df[(full_df["model_name"] == m) & (full_df["error_message"] == 0)]
        scores = valid_rows["coverage_percentage"].values
        
        if len(scores) > 0:
            model_valid_data[m] = scores
            model_means[m] = np.mean(scores)
        else:
            model_valid_data[m] = []
            model_means[m] = -1.0 # 放在最下面
            
    # 按平均覆盖率排序模型
    sorted_models = sorted(unique_models, key=lambda x: model_means.get(x, -1))

    data = [model_valid_data[model] for model in sorted_models]

    # 3. 绘图逻辑
    fig, ax = plt.subplots(figsize=(14, max(8, len(sorted_models) * 0.4))) # 动态调整高度

    # Create horizontal boxplot
    bp = ax.boxplot(
        data,
        labels=sorted_models,
        patch_artist=True,
        vert=False,
        boxprops={"facecolor": "bisque"},
    )

    # Calculate means for display
    means = [model_means[m] for m in sorted_models]
    
    # 4. 计算统计显著性 (Pairwise Intersection with Ref Model)
    
    # 预先获取 Reference Model 的所有记录，方便快速索引
    ref_df = full_df[full_df["model_name"] == ref_model].set_index("record_index")
    
    labels = []
    
    for model, mean in zip(sorted_models, means):
        count_n = len(model_valid_data[model])
        
        if model == ref_model:
             labels.append(f"{model}       (μ={mean:.1f}, N={count_n}, Ref)   ")
             continue
             
        if ref_model not in unique_models:
             labels.append(f"{model}       (μ={mean:.1f}, N={count_n})")
             continue

        # 获取当前模型数据
        curr_df = full_df[full_df["model_name"] == model].set_index("record_index")
        
        # 找到与 Reference Model 的交集 (Pairwise Intersection)
        # 两个模型都必须 error_message == 0
        common_indices = curr_df[curr_df["error_message"] == 0].index.intersection(
            ref_df[ref_df["error_message"] == 0].index
        )
        
        if len(common_indices) < 2:
            labels.append(f"{model}       (μ={mean:.1f}, N={count_n}, p=N/A)")
            continue
            
        # 提取配对数据
        model_scores = curr_df.loc[common_indices, "coverage_percentage"].values
        ref_scores = ref_df.loc[common_indices, "coverage_percentage"].values
        
        # Wilcoxon signed-rank test
        try:
            # alternative='two-sided' is default, use that or 'greater' if testing improvement
            _, p_val = wilcoxon(ref_scores, model_scores)
        except ValueError:
            # e.g. all diffs are zero
            p_val = 1.0

        labels.append(f"{model}       (μ={mean:.1f}, N={count_n}, p={p_val:.2f})")

    ax.set_yticklabels(labels)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(-5, 105)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    
    plt.title(f"Model Comparison (Sorted by Mean Coverage)\nP-values vs {ref_model} (calculated on paired intersection)", pad=20)
    plt.tight_layout()
    
    try:
        plt.savefig(output_plot)
        print(f"图表已保存到: {output_plot}")
    except Exception as e:
        print(f"保存图表失败: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare gold vs pred YAML using field existence comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  模式1 (CSV + JSON文件夹):
    python compare_filed_new.py --mode csv-json --csv data.csv --json-folder ./json_files --gold-column "newAnswer Body" --output results.csv
  
  模式2 (仅CSV, 提取code block):
    python compare_filed_new.py --mode csv --csv data.csv --gold-column "gpt_Generated_Response" --pred-column "gpt_Refined_Response" --output results.csv
  
  模式3 (仅CSV, 直接YAML):
    python compare_filed_new.py --mode csv-raw --csv data.csv --gold-column "gold_col" --pred-column "pred_col" --output results.csv

  模式4 (双CSV混合, Gold提取Block, Pred直接YAML):
    python compare_filed_new.py --mode csv-mixed --gold-csv gold.csv --pred-csv pred.csv --gold-column "gold_col" --pred-column "pred_col" --output results.csv

  模式5 (CSV + JSON 批量文件):
    python compare_filed_new.py --mode csv-batch-json --csv data.csv --json-folder ./json_results --gold-column "newAnswer Body" --output results.csv

  模式6 (绘图对比):
    python compare_filed_new.py --mode plot --plot-data "Base=results_base.csv" "Finetuned=results_ft.csv" --ref-model "Base" --output plot.png
        """
    )
    
    parser.add_argument("--mode", choices=["csv-json", "csv", "csv-raw", "csv-mixed", "csv-batch-json", "plot"], required=True,
                       help="比较模式: csv-json / csv / csv-raw / csv-mixed / csv-batch-json / plot")
    
    # 通用CSV参数
    parser.add_argument("--csv", help="CSV文件路径")
    parser.add_argument("--gold-csv", help="Gold CSV文件路径 (模式4)")
    parser.add_argument("--pred-csv", help="Pred CSV文件路径 (模式4)")

    parser.add_argument("--gold-column", default="newAnswer Body",
                       help="CSV中gold列的列名")
    parser.add_argument("--pred-column", default="gpt_Refined_Response",
                       help="CSV中pred列的列名")
    
    # 模式1和5特有
    parser.add_argument("--json-folder", help="包含JSON文件的文件夹路径（模式1/5需要）")
    
    # 绘图模式参数
    parser.add_argument("--plot-data", nargs="+", help="绘图数据源，格式为 'ModelName=PathToCSV' (模式6)")
    parser.add_argument("--ref-model", help="参考模型名称，用于计算统计显著性 (模式6)")
    
    # 输出参数
    parser.add_argument("--output", help="输出文件路径（CSV或图片）")
    parser.add_argument("--limit", type=int, help="限制处理的记录数")
    parser.add_argument("--model-name", default="manifest", help="模型名称，用于创建文件夹保存最佳 pred YAML（仅 csv 模式）")
    
    args = parser.parse_args()
    
    if args.mode == "csv-json":
        if not args.csv:
            print("错误: 模式1需要 --csv 参数", file=sys.stderr)
            sys.exit(1)
        if not args.json_folder:
            print("错误: 模式1需要 --json-folder 参数", file=sys.stderr)
            sys.exit(1)
        
        process_mode1(
            csv_path=args.csv,
            json_folder=args.json_folder,
            gold_column=args.gold_column,
            output_file=args.output,
            limit=args.limit
        )
    
    elif args.mode == "csv":
        if not args.csv:
            print("错误: 模式2需要 --csv 参数", file=sys.stderr)
            sys.exit(1)
        
        process_mode2(
            csv_path=args.csv,
            gold_column=args.gold_column,
            pred_column=args.pred_column,
            output_file=args.output,
            limit=args.limit,
            model_name=args.model_name
        )
    
    elif args.mode == "csv-raw":
        if not args.csv:
            print("错误: 模式3需要 --csv 参数", file=sys.stderr)
            sys.exit(1)
        
        process_mode_direct(
            csv_path=args.csv,
            gold_column=args.gold_column,
            pred_column=args.pred_column,
            output_file=args.output,
            limit=args.limit
        )

    elif args.mode == "csv-mixed":
        if not args.gold_csv:
            print("错误: 模式4需要 --gold-csv 参数", file=sys.stderr)
            sys.exit(1)
        if not args.pred_csv:
            print("错误: 模式4需要 --pred-csv 参数", file=sys.stderr)
            sys.exit(1)
        
        process_mode4(
            gold_csv=args.gold_csv,
            pred_csv=args.pred_csv,
            gold_column=args.gold_column,
            pred_column=args.pred_column,
            output_file=args.output,
            limit=args.limit
        )
        
    elif args.mode == "csv-batch-json":
        if not args.csv:
            print("错误: 模式csv-batch-json需要 --csv 参数", file=sys.stderr)
            sys.exit(1)
        if not args.json_folder:
            print("错误: 模式csv-batch-json需要 --json-folder 参数", file=sys.stderr)
            sys.exit(1)
            
        process_mode_batch_json(
            csv_path=args.csv,
            json_folder=args.json_folder,
            gold_column=args.gold_column,
            output_file=args.output,
            limit=args.limit
        )

    elif args.mode == "plot":
        if not args.plot_data:
            print("错误: 模式plot需要 --plot-data 参数 (格式 'Model=path.csv')", file=sys.stderr)
            sys.exit(1)
        
        output_plot = args.output if args.output else "comparison_plot.png"
        ref_model = args.ref_model if args.ref_model else ""
        
        process_plot_mode(
            plot_data=args.plot_data,
            ref_model=ref_model,
            output_plot=output_plot
        )

if __name__ == "__main__":
    main()
