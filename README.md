# 受限资源下的铁道知识图片描述

<p align="center">
  <img height="20" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&amp;logoColor=white" />
  <img height="20" src="https://img.shields.io/badge/pytorch-2.1%2B-EE4C2C?logo=pytorch&amp;logoColor=white" />
  <img height="20" src="https://img.shields.io/badge/transformers-4.40%2B-FFD21E?logo=huggingface&amp;logoColor=black" />
  <img height="20" src="https://img.shields.io/badge/pillow-10.0%2B-345995" />
  <img height="20" src="https://img.shields.io/badge/pytest-8.0%2B-0A9EDC?logo=pytest&amp;logoColor=white" />
  <img height="20" src="https://img.shields.io/badge/ruff-0.4%2B-D7FF64?logo=ruff&amp;logoColor=black" />
</p>

这是一个用于论文实验的代码框架，目标是支撑：

- 铁路专用图文数据集构建
- 铁路知识库/术语表约束
- 轻量级图像描述模型训练
- 蒸馏、量化、导出等资源受限部署实验
- 通用 caption 指标与铁路领域指标评估

## 数据格式

训练、验证、测试集使用 JSONL 清单，每行一条样本：

```json
{
  "image": "data/images/train/000001.jpg",
  "caption": "图中可见钢轨、轨枕和扣件，扣件状态正常，未发现异物侵限风险。",
  "equipment": ["钢轨", "轨枕", "扣件"],
  "faults": [],
  "risks": ["无明显风险"],
  "split": "train"
}
```

字段说明：

- `image`: 图片路径，可为相对仓库根目录路径或绝对路径
- `caption`: 人工标注或知识驱动生成的描述
- `equipment`: 图片中出现的铁路设备/部件
- `faults`: 缺陷或异常术语，没有则为空数组
- `risks`: 风险语义标签，例如 `异物侵限风险`、`扣件松动风险`
- `split`: `train`、`val` 或 `test`

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

railcap-build-manifest --input data/raw_annotations.jsonl --output data/train.jsonl
railcap-train --config configs/default.yaml
railcap-evaluate --config configs/default.yaml --predictions outputs/predictions.jsonl
pytest
```

## 目录结构

```text
configs/                  实验配置
data/ontology/            铁路知识库和术语表
src/railcap/data/         数据协议、数据集、清单构建
src/railcap/knowledge/    知识库加载、术语约束、幻觉检查
src/railcap/models/       模型构建、蒸馏、压缩和导出
src/railcap/metrics/      铁路领域指标
src/railcap/train.py      训练入口
src/railcap/evaluate.py   评估入口
tests/                    单元测试
```

## 论文实验建议

核心对比组建议至少包含：

1. 通用 caption 模型直接微调
2. 加铁路术语约束
3. 加知识增强提示或检索
4. 小模型蒸馏
5. 量化/导出后的边缘部署结果

领域指标建议报告：

- 设备术语召回率
- 故障术语 F1
- 风险描述准确率
- 铁路术语幻觉率
- 参数量、显存、延迟、吞吐量

<!-- codex-runtime-notes:start -->

## Runtime Ports And Database Configuration

### Database
- No application database is used. Inputs and outputs are local JSONL, image, ontology, and experiment files.

### Default Ports
- No default web service or database port is defined.

### Notes
- Use local config files under `configs/` and CLI entry points from `pyproject.toml`.

### Source Files Checked
- `pyproject.toml`
- `configs/default.yaml`
- `README.md`

<!-- codex-runtime-notes:end -->
