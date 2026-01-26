# compare_filed_new.py 使用说明

用于比较 Gold YAML 与 Pred YAML 的字段覆盖率评估工具。

## 依赖安装

```bash
pip install pyyaml pandas matplotlib scipy numpy
```

## 命令格式

```bash
python compare_filed_new.py --mode <模式> [参数...]
```

---

## 模式说明

### 模式1: `csv-json` - CSV + JSON 文件夹

从 CSV 提取 gold，从 JSON 文件夹中按索引匹配的 JSON 文件提取 pred。

**参数：**
| 参数 | 必需 | 说明 |
|------|------|------|
| `--csv` | ✅ | CSV 文件路径 |
| `--json-folder` | ✅ | JSON 文件夹路径（包含 0.json, 1.json...） |
| `--gold-column` | ❌ | Gold 列名（默认: `newAnswer Body`） |
| `--output` | ❌ | 输出 CSV 文件路径 |
| `--limit` | ❌ | 限制处理记录数 |

**示例：**
```bash
python compare_filed_new.py \
    --mode csv-json \
    --csv data/input.csv \
    --json-folder data/json_outputs/ \
    --gold-column "newAnswer Body" \
    --output results/comparison_results.csv \
    --limit 100
```

---

### 模式2: `csv` - 单 CSV（提取代码块）

从同一个 CSV 的两列分别提取 gold 和 pred。
- Gold: 提取 ` ```yaml ` 代码块
- Pred: 提取 ` ```yaml:complete ` 或 ` ```yaml: complete ` 代码块

**参数：**
| 参数 | 必需 | 说明 |
|------|------|------|
| `--csv` | ✅ | CSV 文件路径 |
| `--gold-column` | ❌ | Gold 列名（默认: `gpt_Generated_Response`） |
| `--pred-column` | ❌ | Pred 列名（默认: `gpt_Refined_Response`） |
| `--model-name` | ❌ | 模型名称，用于创建 YAML 输出文件夹（默认: `manifest`） |
| `--output` | ❌ | 输出 CSV 文件路径 |
| `--limit` | ❌ | 限制处理记录数 |

**示例：**
```bash
python compare_filed_new.py \
    --mode csv \
    --csv results/gpt4_results.csv \
    --gold-column "newAnswer Body" \
    --pred-column "gpt_Generated_Response" \
    --model-name gpt4_model \
    --output results/gpt4_eval.csv
```

---

### 模式3: `csv-raw` - 单 CSV（直接 YAML）

从同一个 CSV 的两列直接作为 YAML 内容处理（不提取代码块）。

**参数：**
| 参数 | 必需 | 说明 |
|------|------|------|
| `--csv` | ✅ | CSV 文件路径 |
| `--gold-column` | ✅ | Gold 列名 |
| `--pred-column` | ✅ | Pred 列名 |
| `--output` | ❌ | 输出 CSV 文件路径 |
| `--limit` | ❌ | 限制处理记录数 |

**示例：**
```bash
python compare_filed_new.py \
    --mode csv-raw \
    --csv data/yaml_pairs.csv \
    --gold-column "gold_yaml" \
    --pred-column "pred_yaml" \
    --output results/raw_comparison.csv
```

---

### 模式4: `csv-mixed` - 双 CSV 混合

两个 CSV 文件，Gold 提取代码块，Pred 直接作为 YAML。

**参数：**
| 参数 | 必需 | 说明 |
|------|------|------|
| `--gold-csv` | ✅ | Gold CSV 文件路径 |
| `--pred-csv` | ✅ | Pred CSV 文件路径 |
| `--gold-column` | ✅ | Gold 列名 |
| `--pred-column` | ✅ | Pred 列名 |
| `--output` | ❌ | 输出 CSV 文件路径 |
| `--limit` | ❌ | 限制处理记录数 |

**示例：**
```bash
python compare_filed_new.py \
    --mode csv-mixed \
    --gold-csv data/gold_answers.csv \
    --pred-csv data/model_predictions.csv \
    --gold-column "newAnswer Body" \
    --pred-column "yaml_content" \
    --output results/mixed_comparison.csv
```

---

### 模式5: `csv-batch-json` - CSV + 批量 JSON

从 CSV 提取 gold，从多个 JSON 文件（0.json, 1.json...）合并后按顺序匹配 pred。

**参数：**
| 参数 | 必需 | 说明 |
|------|------|------|
| `--csv` | ✅ | CSV 文件路径 |
| `--json-folder` | ✅ | JSON 文件夹路径 |
| `--gold-column` | ❌ | Gold 列名（默认: `newAnswer Body`） |
| `--output` | ❌ | 输出 CSV 文件路径 |
| `--limit` | ❌ | 限制处理记录数 |

**示例：**
```bash
python compare_filed_new.py \
    --mode csv-batch-json \
    --csv data/questions.csv \
    --json-folder data/batch_results/ \
    --gold-column "newAnswer Body" \
    --output results/batch_eval.csv
```

---

### 模式6: `plot` - 绘图对比

根据多个评估结果 CSV 绘制模型对比箱线图，计算统计显著性。

**参数：**
| 参数 | 必需 | 说明 |
|------|------|------|
| `--plot-data` | ✅ | 数据源列表，格式: `ModelName=CSVPath`（可多个） |
| `--ref-model` | ❌ | 参考模型名称（用于计算 p 值） |
| `--output` | ❌ | 输出图片路径（默认: `comparison_plot.png`） |

**示例：**
```bash
python compare_filed_new.py \
    --mode plot \
    --plot-data "GPT4=results/gpt4_eval.csv" "Claude=results/claude_eval.csv" "Gemini=results/gemini_eval.csv" \
    --ref-model "GPT4" \
    --output figures/model_comparison.png
```

---

## 输出说明

### 评估结果 CSV 格式

| 列名 | 说明 |
|------|------|
| `record_index` | 记录索引 |
| `success` | 成功时包含 JSON 格式的比较结果 |
| `failure` | 失败时包含错误信息 |

### success 字段内容

```json
{
  "record_index": 0,
  "total_fields_in_gold": 15,
  "existing_fields_in_pred": 12,
  "coverage_percentage": 80.0,
  "missing_fields_count": 3,
  "missing_fields": "['spec.containers.[0].ports', ...]"
}
```

### 统计输出示例

```
============================================================
比较结果统计
============================================================
成功比较记录数: 93/101
失败记录数: 8
平均覆盖率: 40.06%
最高覆盖率: 100.00%
最低覆盖率: 0.00%

覆盖率分布:
  100%: 21 条 (22.6%)
  90%-100%: 22 条 (23.7%)
  80%-89%: 4 条 (4.3%)
  70%-79%: 6 条 (6.5%)
  60%-69%: 7 条 (7.5%)
  低于60%: 54 条 (58.1%)
```

---

## 常见用法

### 评估单个模型结果
```bash
python compare_filed_new.py \
    --mode csv \
    --csv model_output.csv \
    --gold-column "newAnswer Body" \
    --pred-column "gpt_Generated_Response" \
    --model-name my_model \
    --output my_model_eval.csv
```

### 批量评估多个模型并绘图
```bash
# 1. 评估各模型
python compare_filed_new.py --mode csv --csv gpt4.csv --gold-column "newAnswer Body" --pred-column "gpt_Generated_Response" --model-name gpt4 --output gpt4_eval.csv
python compare_filed_new.py --mode csv --csv claude.csv --gold-column "newAnswer Body" --pred-column "gpt_Generated_Response" --model-name claude --output claude_eval.csv

# 2. 绘制对比图
python compare_filed_new.py --mode plot --plot-data "GPT4=gpt4_eval.csv" "Claude=claude_eval.csv" --ref-model "GPT4" --output comparison.png
```

