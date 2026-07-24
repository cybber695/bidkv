# BidKV 在 SGLang 上完整复现记录

> 日期：2026-07-24 | 作者：BidKV Team

## 一、环境信息

| 项目 | 值 |
|------|-----|
| OS | Ubuntu 22.04 LTS (Docker 容器) |
| GPU | NVIDIA RTX A6000 48GB |
| NVIDIA Driver | 550.144.03 |
| CUDA Driver API | 12.4 |
| Python | 3.11.15 |
| Conda 环境 | `bidkv_sglang` |

## 二、关键依赖版本

| 包名 | 版本 | 说明 |
|------|------|------|
| `sglang` | **0.5.9** | ⚠️ 必须此版本，0.4.10 适配器不兼容 |
| `torch` | 2.7.1 | |
| `flashinfer-python` | 0.2.9rc2 | |
| `flashinfer-cubin` | 0.6.12 | 预编译 CUDA kernel |
| `cuda-toolkit` (conda) | **12.8.1** | conda 环境内安装，CUDA_HOME 指向此处 |
| `huggingface_hub` | 0.36.2 | |
| `transformers` | 4.55.0 | |

### 版本兼容性对照

| SGLang 版本 | 适配器兼容 | 说明 |
|-------------|-----------|------|
| 0.4.10.post2 | ❌ 不兼容 | ScheduleBatch API 不同，`get_kv_stats()` 返回 0 |
| **0.5.9** | ✅ 兼容 | 文档指定版本，`token_to_kv_pool_allocator` 等 API 匹配 |

## 三、环境搭建步骤

### 3.1 安装 CUDA 12.8 工具链（在 conda 环境内）

系统 `/usr/local/cuda` 是 CUDA 13.2（太新，与 12.4 驱动不兼容）。需要在 conda 环境内安装 CUDA 12.8：

```bash
# 在 bidkv_sglang 环境中安装完整 CUDA 12.8.1 工具链
conda install -n bidkv_sglang -c nvidia -c conda-forge cuda-toolkit=12.8.1 -y
```

安装后 CUDA 位置：
- `bin/nvcc` → `/root/miniconda3/envs/bidkv_sglang/bin/nvcc` (12.8.93)
- `include/` → `/root/miniconda3/envs/bidkv_sglang/targets/x86_64-linux/include/`
- `lib/` → `/root/miniconda3/envs/bidkv_sglang/lib/` (libcudart.so)

### 3.2 修复 lib64 符号链接

conda CUDA 工具链将库文件放在 `lib/`，但 flashinfer JIT 编译时链接器找 `lib64/`：

```bash
rm -rf /root/miniconda3/envs/bidkv_sglang/lib64
ln -sf lib /root/miniconda3/envs/bidkv_sglang/lib64
```

### 3.3 升级 SGLang 到 0.5.9

```bash
/root/miniconda3/envs/bidkv_sglang/bin/pip install sglang==0.5.9
```

### 3.4 配置 HuggingFace 镜像和 Token

由于容器无法直连 huggingface.co，使用镜像 `hf-mirror.com`：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=<your_hf_token>  # meta-llama 是 gated 模型
```

### 3.5 配置代理（可选，如容器可直连则跳过）

```bash
export HTTP_PROXY=http://10.14.146.219:7897
export HTTPS_PROXY=http://10.14.146.219:7897
export no_proxy="127.0.0.1,localhost"
```

## 四、代码修复

### 4.1 serve_entry.py 多进程注入修复

**文件**：`src/bidkv/experiments/sglang/serve_entry.py`

**问题**：SGLang 0.5.9 用 `spawn` 方式启动 scheduler 子进程，主进程中 monkey-patch 的 `Scheduler.__init__` 不会被子进程继承。

**修复**：改用 `launch_server` 的 `run_scheduler_process_func` 参数，将 `_bidkv_run_scheduler_process` 传入，让它在子进程内部完成 patch。

```python
# 修复前 (0.4.x 路径)：
_patch_sglang_scheduler(strategy)
launch_server(server_args)

# 修复后 (0.5.x 路径)：
try:
    launch_server(
        server_args,
        run_scheduler_process_func=_bidkv_run_scheduler_process,
    )
except TypeError:
    # 0.4.x fallback
    _patch_sglang_scheduler(strategy)
    launch_server(server_args)
```

### 4.2 scheduler_hook.py 调试增强（可选）

**文件**：`src/bidkv/adapters/sglang/scheduler_hook.py`

在诊断输出中增加 `running_batch`、`kv_allocator`、`waiting_queue` 的实际类型和长度信息，方便排查 API 匹配问题。

## 五、运行实验

### 5.1 完整运行命令

```bash
cd /home/bidkv

export CUDA_HOME=/root/miniconda3/envs/bidkv_sglang
export PATH="/root/miniconda3/envs/bidkv_sglang/bin:$PATH"
export HTTP_PROXY=http://10.14.146.219:7897       # 按需
export HTTPS_PROXY=http://10.14.146.219:7897       # 按需
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=<your_hf_token>

/root/miniconda3/envs/bidkv_sglang/bin/python -m bidkv.experiments.sglang.runner \
  --strategies bidkv \
  --workloads mixed \
  --mixed-rates 3.8 \
  --runs 1 \
  --max-total-tokens 9600 \
  --output-dir results/sglang_test_run \
  --traces-dir experiments/vllm/traces
```

### 5.2 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--strategies` | bidkv | BidKV 完整 bid pipeline |
| `--workloads` | mixed | ShareGPT mixed 工作负载 |
| `--mixed-rates` | 3.8 | 3.8 req/s（关键 rate） |
| `--runs` | 1 | 单次运行 |
| `--max-total-tokens` | 9600 | KV cache 上限 = 600 blocks × 16 |
| `--traces-dir` | experiments/vllm/traces | 冻结 trace（seed=42） |

> **注意**：不需要 `--disable-cuda-graph`。CUDA 12.8 工具链可正常编译 flashinfer kernel 和捕获 CUDA graph。

### 5.3 后台运行（推荐）

```bash
nohup bash -c '...上述命令...' > results/sglang_test_run/run.log 2>&1 &
```

## 六、实验结果

### 6.1 主指标

| 指标 | 值 | 论文 v10 参考 |
|------|-----|-------------|
| **TTFT P50** | 87.4 ms | — |
| **TTFT P95** | 262.1 ms | 359 ms ✅ |
| **TTFT P99** | 433.7 ms | — |
| **SLO(300ms)** | 96.9% | 92.4% ✅ |
| **SLO(500ms)** | 99.2% | — |
| **SLO(1000ms)** | 100.0% | — |
| **TPOT P50** | 36.5 ms | — |
| **TPOT P95** | 57.3 ms | — |
| **TPOT P99** | 81.7 ms | — |
| **Throughput** | 3.68 req/s | — |
| **成功/失败** | 1000 / 0 | — |
| **运行时长** | 271.4 s | — |

### 6.2 TTFT 完整分布

```
P1:    52.9ms    P25:   77.4ms    P75:  102.0ms    P99:  437.0ms
P5:    66.5ms    P50:   87.4ms    P90:  180.4ms    P100: 835.6ms
P10:   69.6ms    P95:  262.1ms
```

### 6.3 TPOT 完整分布

```
P1:   29.8ms    P25:  33.6ms    P75:  40.6ms    P99:   83.1ms
P5:   31.1ms    P50:  36.5ms    P90:  48.1ms    P100: 317.5ms
P10:  31.9ms    P95:  57.3ms
```

### 6.4 结果文件

```
results/sglang_test_run/sglang__bidkv__mixed__rate3.8__run0.json  (1.16 MB)
```

### 6.5 BidKV Solver 活动确认

服务器日志中可见 GreedyBidSolver 正常工作：

```
GreedyBidSolver: accepted 11 bids, freed=3987 tokens, delta=11.0000, elapsed_ms=0.01
GreedyBidSolver: accepted 8 bids, freed=2342 tokens, delta=8.0000, elapsed_ms=0.01
GreedyBidSolver: accepted 9 bids, freed=551 tokens, delta=9.0000, elapsed_ms=0.01
```

## 七、故障排查记录

### 7.1 问题：TTFT P95 = 4604ms vs 论文 359ms（差 13 倍）

**现象**：SGLang 日志显示 `#running-req: 19`，但 BidKV 诊断显示 `running=0, kv=0/9600`

**根因**：SGLang 0.4.10 API 与适配器不匹配
- `_get_token_to_kv_pool()` 找 `scheduler.token_to_kv_pool_allocator`，但 0.4.10 scheduler 没有此属性
- KV pool 在 `batch.token_to_kv_pool_allocator` 上
- 请求列表在 `batch.decoding_reqs` 而非 `batch.reqs`

**解决**：升级到 SGLang 0.5.9（文档指定版本）

### 7.2 问题：SGLang 0.5.9 中 hook 被调用但 running=0

**现象**：诊断日志 272K+ 次调用，但 running 始终为 0

**根因**：`serve_entry.py` 只 patch 了主进程的 `Scheduler.__init__`，但 0.5.9 的 scheduler 在 `spawn` 子进程中运行，patch 不继承

**解决**：修改 `serve_entry.py`，通过 `launch_server(run_scheduler_process_func=_bidkv_run_scheduler_process)` 在子进程中注入

### 7.3 问题：flashinfer JIT 编译失败

**错误**：`fatal error: nv/target: No such file or directory`

**根因**：系统 CUDA 13.2 头文件与 12.4 驱动不兼容

**解决**：在 conda 环境中安装 CUDA 12.8.1 工具链，设置 `CUDA_HOME=/root/miniconda3/envs/bidkv_sglang`

### 7.4 问题：链接时找不到 libcudart

**错误**：`/usr/bin/ld: cannot find -lcudart`

**根因**：conda CUDA 库在 `lib/` 但链接器找 `lib64/`

**解决**：`ln -sf lib /root/miniconda3/envs/bidkv_sglang/lib64`

### 7.5 问题：huggingface.co 不可达 + gated 模型 403

**解决**：使用 `hf-mirror.com` 镜像 + `HF_TOKEN`

## 八、已知遗留问题

1. **adapter_metrics 为空**：结果 JSON 中 `adapter_metrics` 字段为空 `{}`，不影响延迟/吞吐数据，但 eviction 统计不可用。推测是 SGLang 子进程间 metric 传递路径问题。
2. **scheduler_hook.py 中的调试代码**：为排查问题而增加的 DEBUG 诊断输出，正式运行前建议移除或降级到 `logger.debug`。
3. **serve_entry.py 0.4.x fallback**：保留了 `TypeError` fallback 以兼容旧版，如果确认只用 0.5.x 可考虑移除。

## 九、环境变量速查

| 变量 | 必须? | 值 | 说明 |
|------|-------|-----|------|
| `CUDA_HOME` | ✅ | `/root/miniconda3/envs/bidkv_sglang` | conda 环境内的 CUDA 12.8 |
| `PATH` | ✅ | 包含 `.../bin` | 确保使用 conda 环境的 nvcc |
| `HF_ENDPOINT` | ✅ | `https://hf-mirror.com` | HuggingFace 镜像 |
| `HF_TOKEN` | ✅ | `hf_...` | gated 模型认证 |
| `HTTP_PROXY` | 按需 | `http://10.14.146.219:7897` | 代理 |
| `HTTPS_PROXY` | 按需 | 同上 | 代理 |
| `no_proxy` | ✅ | `127.0.0.1,localhost` | 本地不走代理 |
| `HF_HUB_OFFLINE` | ❌ | 不设 | 会导致 huggingface_hub 0.36.2 报错 |
| `TRANSFORMERS_OFFLINE` | ❌ | 不设 | 同上，会触发 HF_HUB_OFFLINE |

## 十、结果复现验证命令

```bash
# 读取结果文件并计算指标
python -c "
import json
d = json.load(open('results/sglang_test_run/sglang__bidkv__mixed__rate3.8__run0.json'))
ok = [r for r in d['request_results'] if not r.get('error')]
ttft = sorted([r['ttft_ms'] for r in ok])
def pct(data, p): return data[min(int(len(data)*p/100), len(data)-1)]

print(f'TTFT P50: {pct(ttft,50):.0f}ms  P95: {pct(ttft,95):.0f}ms')
print(f'SLO(300ms): {sum(1 for t in ttft if t<=300)/len(ttft)*100:.1f}%')
print(f'Throughput: {d[\"summary\"][\"throughput_rps\"]:.2f} rps')
print(f'Requests: {len(ok)}/{len(d[\"request_results\"])}')
"
```
