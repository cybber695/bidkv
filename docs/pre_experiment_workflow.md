# BidKV 前置实验完整流程

## 概览

前置实验包含三个阶段的准备工作，目的是在为正式实验矩阵（5 策略 × 2 工作负载 × 3 速率 × 3 重复）标定参数并冻结可复现 trace。

| 阶段 | 名称 | 目的 |
|------|------|------|
| Phase 1 | 数据集 Tokenization | 生成预 tokenize 的 JSONL pool 文件 |
| Phase 2 | Pilot 校准 | 找每个工作负载的有效压力区间 (rate_low/mid/high) |
| Phase 3 | 冻结 Formal Trace | 生成正式实验用的完整请求 trace (seed=42, 100% 请求数) |

---

## Phase 1: 数据集 Tokenization（数据预处理）

### 1.1 数据集准备

将 ShareGPT 原始数据集 (ShareGPT_Vicuna_unfiltered) 下载并放置到 `data/` 目录下。

输入格式：ShareGPT JSON 数组，每条包含 `conversations` 字段（多轮对话）。

### 1.2 Tokenization

使用目标模型的 tokenizer 分别生成两类 token 池：

```bash
# mixed 工作负载 — 平均 prompt 长度 ~500 tokens（随机采样）
# long_context 工作负载 — 筛选 prompt > 2048 tokens 的记录
# 输出分别保存为 data/sharegpt_mixed_pool.jsonl 和 data/sharegpt_long_pool.jsonl
```

每行 JSON 格式：
```json
{"id": "xxx", "prompt": "...", "output": "...", "prompt_tokens": 512, "output_tokens": 128}
```

### 1.3 验证

```bash
# 检查 pool 文件是否正确生成
wc -l data/sharegpt_mixed_pool.jsonl data/sharegpt_long_pool.jsonl
```

- `sharegpt_mixed_pool.jsonl` 应包含 ≥1000 条记录
- `sharegpt_long_pool.jsonl` 应包含 ≥500 条记录

---

## Phase 2: Pilot 校准（找有效压力区间）

### 2.1 协议说明

根据论文 §3 [2-1] 到 [2-3]：
- 使用 2 个代表性策略：`preempt-evict`（无干预基线）和 `largest-first`（简单 KV 驱逐）
- 使用 50% 请求数 (seed=99)，以节省 GPU 时间
- 从 0.5 req/s 开始，每次乘以 1.5，逐步增加请求速率
- 当 OOM 占比 > 20% 或 timeout 占比 > 20% 时停止该策略的扫描
- 当两个策略都停止时，结束该工作负载的扫描

### 2.2 运行 Pilot 校准脚本

```bash
python -m bidkv.experiments.vllm.pilot_calibration \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.85 \
    --output-dir results/pilot \
    --data-dir data \
    --port 8000
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | 模型名称或路径。默认从 `BIDKV_MODEL` 环境变量读取，未设置时使用此值（见 `common/model.py`） |
| `--gpu-memory-utilization` | 0.85 | GPU 显存利用率。§3 [2-5] 可能需要下调到 0.80 或 0.75 |
| `--output-dir` | `results/pilot` | Pilot 结果输出目录 |
| `--data-dir` | `data` | Tokenized pool 文件目录 |
| `--port` | 8000 | vLLM 服务端口 |

**速率扫描序列：** 0.5, 0.75, 1.13, 1.69, 2.53, 3.8, 5.7, 8.54, 12.81, 19.22, 28.83, 43.24, 64.87, 97.3, 145.95

**停止条件：** 某一策略的 OOM 占比 > 20% 或 timeout 占比 > 20%。

### 2.3 分析 Pilot 报告

```bash
# 查看 pilot 结果
cat results/pilot/pilot_report.json
```

从报告中筛选每个工作负载的 `rate_low` / `rate_mid` / `rate_high`：
- **rate_low**: 轻微 KV 压力，OOM/timeout < 5%
- **rate_mid**: 中等 KV 压力，驱逐事件开始发生
- **rate_high**: 高 KV 压力，系统接近饱和但未崩溃

### 2.4 冻结校准速率（已冻结，当前值）

经 Issue #055 校准后，当前冻结的速率值为：

| 工作负载 | rate_low | rate_mid | rate_high | 校准依据 |
|----------|----------|----------|-----------|----------|
| mixed | 2.0 req/s | 3.8 req/s | 5.7 req/s | 吞吐量: 2.06→3.32→3.84（饱和），P99 TTFT: 356→450→440ms |
| long_context | 0.35 req/s | 0.5 req/s | 0.7 req/s | 吞吐量: 0.50→0.63→0.67（饱和），P99 TTFT: 3.2k→4.9k→10kms |

**⚠️ RULE RATE-FREEZE:** 校准后的速率值写入 `src/bidkv/experiments/vllm/config.py` 的 `WORKLOAD_REQUEST_RATES` 字典，冻结后不得基于策略表现调整。

```python
# src/bidkv/experiments/vllm/config.py:48-51
WORKLOAD_REQUEST_RATES: dict[str, tuple[float, ...]] = {
    WORKLOAD_MIXED: (2.0, 3.8, 5.7),
    WORKLOAD_LONG_CONTEXT: (0.35, 0.5, 0.7),
}
```

---

## Phase 3: 冻结 Formal Trace

### 3.1 生成正式实验 Trace

使用校准后的速率，生成正式实验所需要的完整请求 trace（seed=42, 100% 请求数）。

```bash
# 使用冻结速率生成 formal traces
python -m bidkv.experiments.vllm.freeze_traces \
    --mode formal \
    --use-frozen-rates \
    --output-dir experiments/vllm/traces \
    --data-dir data
```

**参数说明：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `--mode` | `formal` | 使用 seed=42，完整请求数（mixed=1000, long_context=500） |
| `--use-frozen-rates` | - | 从 `WORKLOAD_REQUEST_RATES` 读取校准速率 |
| `--output-dir` | `experiments/vllm/traces` | Trace 输出目录 |
| `--data-dir` | `data` | Tokenized pool 文件目录 |

**生成的 trace 文件：**

| 文件名 | 工作负载 | 速率 | 请求数 | SHA-256 (前12位) |
|--------|----------|------|--------|-------------------|
| `mixed_rate2.0.json` | mixed | 2.0 req/s | 1000 | `6221f218b1a5` |
| `mixed_rate3.8.json` | mixed | 3.8 req/s | 1000 | `67f67193b7e6` |
| `mixed_rate5.7.json` | mixed | 5.7 req/s | 1000 | `c1020d5edaf3` |
| `long_rate0.35.json` | long_context | 0.35 req/s | 500 | `37ec72f2151b` |
| `long_rate0.5.json` | long_context | 0.5 req/s | 500 | `1480e30b3175` |
| `long_rate0.7.json` | long_context | 0.7 req/s | 500 | `b4febee0fd36` |

### 3.2 验证 Trace 可复现性（可选）

```bash
python -m bidkv.experiments.vllm.freeze_traces \
    --mode verify \
    --use-frozen-rates \
    --output-dir experiments/vllm/traces \
    --data-dir data
```

此命令重新生成所有 trace，对比 SHA-256 哈希值，验证可复现性。所有哈希匹配则输出 "PASSED"。

### 3.3 Pilot 模式的 Trace（校准阶段已使用）

Pilot 校准脚本会自动生成 pilot mode traces（`--mode pilot`, seed=99, 50% 请求数），无需手动执行此命令。但如果需要独立生成 pilot traces：

```bash
# 示例：为手动指定的速率生成 pilot traces
python -m bidkv.experiments.vllm.freeze_traces \
    --mode pilot \
    --rates 2.0,3.8,5.7 \
    --output-dir experiments/vllm/traces/pilot \
    --data-dir data
```

---

## Phase 4: 运行正式实验

前置实验完成后，即可运行正式实验矩阵。

### 4.1 快速验证（Smoke Test）

先用一个 com​​bination 验证流程是否正确：

```bash
python -m bidkv.experiments.vllm.runner \
    --strategies preempt-evict \
    --workloads mixed \
    --runs 1 \
    --mixed-rates 2.0 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.50 \
    --output-dir results/vllm_smoke/
```

### 4.2 完整实验

```bash
# 完整 5 策略 × 2 工作负载 × 3 速率 × 3 重复 = 最大 90 次运行
python -m bidkv.experiments.vllm.runner \
    --strategies preempt-evict,preempt-evict-sjf,static-random,largest-first,bidkv \
    --workloads mixed,long_context \
    --runs 3 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.50 \
    --output-dir results/vllm_$(date +%Y%m%d)/ \
    --resume
```

**关键参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--strategies` | 全部 5 个 | 要运行的策略 |
| `--runs` | 3 | 每个 (strategy, workload, rate) 组合的重复次数 |
| `--gpu-memory-utilization` | 0.50 | GPU 显存超分参数（正式实验用 0.50 而非 pilot 的 0.85） |
| `--max-model-len` | 8192 | 最大模型输入长度 |
| `--resume` | false | 断点续跑，跳过已有结果文件的 runs |

### 4.3 分析实验结果

```bash
# vLLM 分析
python -m bidkv.experiments.vllm.analysis \
    --results-dir results/vllm_YYYYMMDD/

# SGLang 跨框架分析
python -m bidkv.experiments.sglang.analysis \
    --vllm-dir results/vllm_YYYYMMDD/ \
    --sglang-dir results/sglang_YYYYMMDD/
```

---

## 快速检查清单

执行完整前置实验前，按顺序检查：

- [ ] **GPU 环境**: `nvidia-smi` 正常，有足够显存
- [ ] **模型已下载**: `meta-llama/Llama-3.1-8B-Instruct` 或设置 `BIDKV_MODEL` 指向本地路径
- [ ] **Phase 1 完成**: `data/sharegpt_mixed_pool.jsonl` 和 `data/sharegpt_long_pool.jsonl` 存在且行数满足要求
- [ ] **BidKV 包安装**: `pip install -e ".[dev]"` 成功
- [ ] **环境变量**: 如有必要设置 `HF_HUB_OFFLINE=1`（离线环境）
- [ ] **Phase 2 完成**: `results/pilot/pilot_report.json` 已生成，速率已校准
- [ ] **Phase 3 完成**: `experiments/vllm/traces/manifest.json` 存在，6 个 trace 文件 SHA-256 已验证
- [ ] **Smoke test 通过**: 单 run 实验成功完成

---

## 文件结构

```
bidkv/
├── data/
│   ├── sharegpt_mixed_pool.jsonl       # Phase 1: Mixed pool
│   └── sharegpt_long_pool.jsonl        # Phase 1: Long-context pool
├── experiments/vllm/traces/
│   ├── manifest.json                   # Phase 3: Trace manifest
│   ├── mixed_rate2.0.json              # Phase 3: Mixed @ 2.0 req/s
│   ├── mixed_rate3.8.json              # Phase 3: Mixed @ 3.8 req/s
│   ├── mixed_rate5.7.json              # Phase 3: Mixed @ 5.7 req/s
│   ├── long_rate0.35.json              # Phase 3: Long-context @ 0.35 req/s
│   ├── long_rate0.5.json               # Phase 3: Long-context @ 0.5 req/s
│   └── long_rate0.7.json               # Phase 3: Long-context @ 0.7 req/s
├── results/
│   ├── pilot/
│   │   └── pilot_report.json           # Phase 2: Pilot 校准报告
│   └── vllm_YYYYMMDD/                  # Phase 4: 正式实验结果
│       ├── preempt-evict__mixed__rate2.0__r0.json
│       ├── ...
│       └── candidate_consistency_report.json
└── src/bidkv/experiments/vllm/
    ├── config.py                        # ⚠️ RULE RATE-FREEZE
    ├── freeze_traces.py                 # Phase 2 & 3: Trace 冻结脚本
    ├── pilot_calibration.py             # Phase 2: Pilot 校准脚本
    ├── runner.py                        # Phase 4: 正式实验编排
    └── analysis.py                      # Phase 4: 结果分析
```
