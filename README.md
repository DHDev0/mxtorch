# mxtorch

**Sub-byte MX quantization for PyTorch with Triton GPU kernels**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![ROCm / CUDA](https://img.shields.io/badge/GPU-ROCm%20%7C%20CUDA-green.svg)](https://rocm.docs.amd.com/)

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
  - [Configuration](#configuration)
  - [Type System — 288 dtype codes](#type-system)
  - [mx_tensor](#mx_tensor)
  - [Quantization Methods](#quantization-methods)
  - [Neural Network Modules](#neural-network-modules)
  - [Gradient / Training](#gradient--training)
  - [Optimizers](#optimizers)
  - [Model-Level Operations](#model-level-operations)
  - [Analysis Tools](#analysis-tools)
  - [Distributed Training (DDP / FSDP)](#distributed-training)
  - [KV Cache Quantization](#kv-cache-quantization)
  - [Custom Kernel Registration](#custom-kernel-registration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Overview

mxtorch implements **MX (Microscaling) quantization** for PyTorch. Values are *actually bit-packed* — not simulated — so `int4d` tensors consume exactly half the bytes of `int8d`.

### What's inside

| Feature | Detail |
|---------|--------|
| **288 data types** | `{int\|float}{1-32}{d\|u}{variant}` — see [Type System](#type-system) |
| **Real bit packing** | int4d packs 2 values per byte; int2d packs 4; int1d packs 8 |
| **True IEEE grids** | float4d = FP4 E2M1, float8d = FP8 E4M3, float16d = IEEE float16 |
| **Triton kernels** | INT8/INT4/INT2/INT1 GEMM, INT4/INT2 qadd, stochastic quant, Hadamard |
| **torch.Tensor subclass** | `mx_tensor` IS a Tensor — works with `isinstance`, `.device`, `.shape` |
| **STE training** | Straight-through estimator via `_QGemmSTE`, `_QAddSTE` |
| **nn.Module drop-ins** | `mx_linear`, `mx_lstm`, `mx_gru`, `mx_conv2d`, `mx_multihead_attention`, … |
| **FSDP compatible** | `__tensor_flatten__` / `__tensor_unflatten__` implemented |
| **DDP compatible** | `install_ddp_hooks` for gradient packing |

---

## Installation

```bash
git clone https://github.com/DHDev0/mxtorch.git
cd mxtorch

# GPU (CUDA or ROCm)
pip install torch triton

# Verify
python -c "import mxtorch as mxt; mxt.mx_info.print_module_info()"
```

**Requirements:** Python 3.8+, PyTorch 2.0+, Triton ≥ 2.1 (for GPU kernels)

---

## Quick Start

```python
import mxtorch as mxt
import torch, torch.nn as nn

# ── Tensor quantization ──────────────────────────────────────────────────────
x = torch.randn(512, 512)

q8  = mxt.mx_tensor.quantize(x, mxt.int8d,   block=128)   # 4x compression
q4  = mxt.mx_tensor.quantize(x, mxt.int4d,   block=128)   # 8x compression
qf8 = mxt.mx_tensor.quantize(x, mxt.float8d, block=128)   # 4x, FP8 E4M3 grid
q4h = mxt.mx_tensor.quantize(x, mxt.int4dh,  block=128)   # 8x + Hadamard rotation

print(f"int8d  compression: {q8.compression_ratio:.1f}x")    # 3.9x (with scale overhead)
print(f"int4d  compression: {q4.compression_ratio:.1f}x")    # 7.5x
print(f"SNR int8d: {mxt.snr(x, 'int8d'):.1f} dB")           # ~44 dB
print(f"SNR int4d: {mxt.snr(x, 'int4d'):.1f} dB")           # ~19 dB
print(f"SNR float8d: {mxt.snr(x, 'float8d'):.1f} dB")       # ~32 dB

# Dequantize at the PyTorch boundary (e.g., before loss)
x_restored = q8.dequantize()

# ── Arithmetic stays quantized ───────────────────────────────────────────────
a = mxt.mx_tensor.quantize(torch.randn(256), mxt.int8d)
b = mxt.mx_tensor.quantize(torch.randn(256), mxt.int8d)
c = a + b   # → mx_tensor (Q7 integer add, zero fp32 per-element)
d = a * b   # → mx_tensor (integer multiply + shift)
e = -a      # → mx_tensor (negate codes, scale unchanged)
f = a.abs() # → mx_tensor (abs codes, scale unchanged)
g = a.relu()# → mx_tensor (clamp codes≥0)
h = a.gelu()# → mx_tensor (float GELU → requantize)

A = mxt.mx_tensor.quantize(torch.randn(64, 128), mxt.int8d)
B = mxt.mx_tensor.quantize(torch.randn(128, 64), mxt.int8d)
C = A @ B   # → mx_tensor (INT8 GEMM kernel, BM*BN block tiles)

# ── Model quantization ───────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128)
)

# Method 1: to_mx
model = mxt.to_mx(model, "int8d", block_size=128)

# Method 2: patched .to()
model = model.to("int4d")
model = model.to(torch.dtype("float8d"))

# Method 3: mixed precision per layer
model = mxt.to_mx(model, {"0": "int4d", "2": "int8d"})

# ── Training ─────────────────────────────────────────────────────────────────
optimizer = mxt.mx_adam_w(model.parameters(), state_dtype="int8d")

for x_batch, y_batch in dataloader:
    out = model(x_batch)                    # weight-only quant, fp32 compute
    if isinstance(out, mxt.mx_tensor):
        out = out.dequantize()
    loss = F.cross_entropy(out, y_batch)
    loss.backward()                         # STE gradients flow through
    optimizer.step()                        # states stored at int8
    optimizer.zero_grad()
```

---

## Core Concepts

### MX Data Types

```
{kind}{bits}{mode}{variant}

kind    ∈ {int, float}
bits    ∈ {1, 2, 4, 8, 16, 32}          (standard widths)
mode    ∈ {d, u}                         d=down/saturating, u=up/zero-padded
variant ∈ {"", "h", "v", "s", "b", "2"} see table below
```

#### Variants

| Suffix | Name | Description | Best for |
|--------|------|-------------|----------|
| *(none)* | Base | Block-wise absmax quantization | General inference |
| `h` | Hadamard | QuIP# rotation before quant (reduces outlier impact) | Weights with outliers |
| `v` | Vector-wise | Per-row/column absmax | Attention key/value |
| `s` | Stochastic | Stochastic rounding (zero-mean error) | Training |
| `b` | Boolean | Binary clamp to {0, 1} | 1-bit networks |
| `2` | Double-precision | N-bit storage + fp16 GEMM accumulation | Training quality |

#### Float vs Int at the same bit width

Float dtypes use an IEEE exponential grid (more points near zero), integers use a uniform grid. For Gaussian weight distributions:

| Dtype | Bits | SNR (N(0,1)) | Grid type |
|-------|------|-------------|-----------|
| `int2d` | 2 | 2.3 dB | Uniform |
| `int4d` | 4 | 18.7 dB | Uniform |
| `float4d` | 4 | **19.3 dB** | FP4 E2M1 log-scale |
| `int8d` | 8 | **43.9 dB** | Uniform |
| `float8d` | 8 | 32.0 dB | FP8 E4M3 log-scale |
| `float16d` | 16 | ~96 dB | IEEE float16 |

> **Note:** `int8d` beats `float8d` on Gaussian weights because uniform quantization is optimal for Gaussian distributions. Float formats win for highly non-uniform inputs (very sparse gradients, activations with outliers).

### Block-wise Quantization

Each block of `block` consecutive elements shares one `float32` scale. Smaller blocks = higher accuracy, more scale overhead.

```python
q = mxt.mx_tensor.quantize(x, mxt.int8d, block=128)   # 1 scale per 128 elements
# Scale overhead: n/128 × 4 bytes. At block=128: ~3.1% overhead at int8.
```

Actual compression ratios at 4096×4096 (16.7M elements, block=128):

| Dtype | Packed MB | Scales MB | Total | vs fp32 | vs bf16 |
|-------|-----------|-----------|-------|---------|---------|
| int1d | 2.10 | 0.52 | 2.62 | **25.6x** | 12.8x |
| int2d | 4.19 | 0.52 | 4.72 | **14.2x** | 7.1x |
| int4d | 8.39 | 0.52 | 8.91 | **7.5x** | 3.8x |
| int8d | 16.78 | 0.52 | 17.30 | **3.9x** | 1.9x |
| float4d | 8.39 | 0.52 | 8.91 | **7.5x** | 3.8x |
| float8d | 16.78 | 0.52 | 17.30 | **3.9x** | 1.9x |

### Double Quantization

Quantize the block scales themselves to save an additional ~0.4 bits/parameter:

```python
q  = mxt.double_quantize(weight, "int4d", block=64, super_block=256)
# int4d: 7.5x → 7.8x compression
# int8d: 3.9x → 4.0x compression

q.dequantize()           # → float32
q.nbytes()               # total bytes: q_data + q_scales (int8) + ss_scale (float32)
q.compression_vs_fp32()  # float
```

---

## API Reference

### Configuration

```python
# Global settings (immutable dataclass — use override for temporary changes)
config = mxt.mx_config.current()
# mx_config(block_size=128, strict=False, debug=False, verbose=False,
#           default_dtype='int4d', cache_kernels=True, max_autotune=True)

# Temporary override
with mxt.mx_config.override(block_size=64, strict=True):
    q = mxt.mx_tensor.quantize(x, mxt.int4d)

# Environment variables
# MX_DEBUG=1            verbose debug logging
# MX_DEBUG_VERBOSE=1    include stack traces
# MX_STRICT=1           raise errors instead of fp32 fallback
```

### Type System

```python
# Resolve dtype by name
dt = mxt.get_mx_dtype("int4d")
dt.name             # "int4d"
dt.bits             # 4
dt.is_float         # False
dt.is_int           # True
dt.pack_ratio       # 2 (values per byte)
dt.compression_vs_fp32  # 8.0 (storage only, excl. scales)
dt.is_stochastic    # False
dt.is_hadamard      # False ("h" variant)
dt.is_bool          # False ("b" variant)
dt.is_double_prec   # False ("2" variant)

# Dtypes are singletons
assert mxt.get_mx_dtype("int4d") is mxt.int4d

# Via torch.dtype() compatibility
torch.dtype("int4d")   # → mx_dtype_proxy
model.to(torch.dtype("int4d"))

# Quality comparison
print(mxt.compare_dtypes(weight, ["int2d", "int4d", "int8d", "float4d", "float8d"]))
```

### mx_tensor

```python
# ── Creation ────────────────────────────────────────────────────────────────
q = mxt.mx_tensor.quantize(x, mxt.int4d, block=128)

# ── Properties ──────────────────────────────────────────────────────────────
q.shape             # logical shape (same as x.shape)
q.device            # cuda:0 / cpu
q._mx_dtype         # mx_dtype object
q._mx_scales        # float32 per-block scales [nb]
q._mx_packed        # int8 raw packed storage
q._mx_block         # block size
q.nbytes_packed     # packed + scales bytes
q.compression_ratio # fp32_bytes / packed_bytes

# ── Dequantize ──────────────────────────────────────────────────────────────
x_f = q.dequantize()   # → float32, original shape

# ── Device transfer ──────────────────────────────────────────────────────────
q_gpu  = q.cuda()
q_cpu  = q_gpu.cpu()
q_gpu2 = q.cuda(1)      # specific device
q_same = q.to("cuda:0")

# ── Re-quantize to different dtype ──────────────────────────────────────────
q8  = q.to("int8d")     # → new mx_tensor at int8d
qf4 = q.to(mxt.float4d)

# ── Arithmetic ───────────────────────────────────────────────────────────────
c = a + b    # INT8: Q7 integer rescale add; FP8: IEEE bit-field add
c = a - b
c = a * b    # INT8: integer multiply + /127 shift
c = -a       # negate codes, scale unchanged (INT8: one vectorized op)
c = a.abs()  # abs codes, scale unchanged
c = A @ B    # GEMM kernel (INT8 dp4a, FP8 decode→fp16 dot)

# ── Activations ──────────────────────────────────────────────────────────────
r = q.relu()   # INT8: packed.clamp(min=0). Float: dequant→relu→requant
g = q.gelu()   # Float GELU applied to dequantized values, then requantized
s = q.softmax(dim=-1)  # → dequantize + softmax (softmax needs float)

# ── Shape ops ───────────────────────────────────────────────────────────────
q2 = q.reshape(256, -1)
q2 = q.t()          # transpose → requantize
q2 = q.clone()
q2 = q.contiguous()
```

### Quantization Methods

```python
# ── Standard block-wise absmax ───────────────────────────────────────────────
q  = mxt.mx_quantize(x, "int8d", block=128)
q  = mxt.mx_tensor.quantize(x, mxt.int8d, block=128)

# ── Hadamard rotation (QuIP# style, reduces outlier impact) ──────────────────
rot_matrix, q_had = mxt.hadamard_quantize(x, "int4d", block=64)

# ── Vector-wise (per-row/col) ─────────────────────────────────────────────────
codes, scales = mxt.vector_quantize(x, "int8d", axis=1)  # per-row
x_r = mxt.vector_dequantize(codes, scales, axis=1)

# ── Stochastic rounding (training, zero-mean error) ──────────────────────────
q = mxt.stochastic_mx_quantize(x, "int8ds", block=128)

# ── Double quantization (QLoRA style) ────────────────────────────────────────
dq = mxt.double_quantize(x, "int4d", block=64, super_block=256)
x_r = dq.dequantize()
print(f"Total bytes: {dq.nbytes()}")
print(f"Compression: {dq.compression_vs_fp32():.1f}x vs fp32")

# ── NF4 (bitsandbytes style) ──────────────────────────────────────────────────
q_nf4 = mxt.nf4_quantize(x, block=64)
x_r   = mxt.nf4_dequantize(q_nf4)

# ── GPTQ / AWQ ──────────────────────────────────────────────────────────────
q_gptq = mxt.gptq_quantize(weight, "int4d", block=128, groupsize=128)
q_awq  = mxt.awq_quantize(weight,  "int4d", block=128, alpha=0.5)

# ── FP4 (raw 4-bit float storage) ────────────────────────────────────────────
codes, scales, n = mxt.fp4_quantize(x, block=64)

# ── Quality measurement ──────────────────────────────────────────────────────
snr_db  = mxt.snr(x, "int8d", block=128)
rmse    = mxt.quantization_error(x, "int8d", block=128, metric="rmse")
mae     = mxt.quantization_error(x, "int8d", block=128, metric="mae")
print(mxt.compare_dtypes(x, ["int2d", "int4d", "int8d", "float8d"]))
```

### Neural Network Modules

All `mx_*` modules are drop-in replacements for their `nn.*` counterparts:

```python
# ── Linear ───────────────────────────────────────────────────────────────────
# Weights stored packed at MX precision; inference uses weight-only dequant
lin = mxt.mx_linear.from_linear(nn.Linear(512, 256), mxt.int8d, block=128)
out = lin(x)   # dequantize weight → F.linear(x_f, w_f) — fast weight-only path

# ── LSTM ─────────────────────────────────────────────────────────────────────
# All 8 gate weight matrices packed at MX precision
lstm = mxt.mx_lstm.from_lstm(nn.LSTM(256, 512, batch_first=True),
                              mx_dtype=mxt.int8d, block=128)
out, (h_n, c_n) = lstm(x)   # x: [B, T, 256]

# ── GRU ─────────────────────────────────────────────────────────────────────
gru = mxt.mx_gru.from_gru_cell(weight_ih, weight_hh, bias_ih, bias_hh,
                                mx_dtype=mxt.int4d)

# ── Attention ────────────────────────────────────────────────────────────────
mha = mxt.mx_multihead_attention.from_mha(
    nn.MultiheadAttention(512, 8, batch_first=True), mxt.int8d, block=128)
out, attn = mha(query, key, value)

# ── Convolutions ─────────────────────────────────────────────────────────────
conv = mxt.mx_conv2d.from_conv2d(nn.Conv2d(32, 64, 3), mxt.int8d)
dconv = mxt.mx_conv_transpose2d.from_conv_transpose2d(
    nn.ConvTranspose2d(64, 32, 4, stride=2), mxt.int4d)

# ── Normalization ────────────────────────────────────────────────────────────
ln = mxt.mx_layer_norm.from_layer_norm(nn.LayerNorm(512), mxt.int8d)
rms = mxt.mx_rms_norm.from_rms_norm(rms_norm_layer, mxt.int8d)
bn = mxt.mx_batch_norm2d.from_batch_norm(nn.BatchNorm2d(64), mxt.int8d)

# ── Embeddings ───────────────────────────────────────────────────────────────
emb = mxt.mx_embedding.from_embedding(nn.Embedding(50000, 512), mxt.int4d)

# ── Transformer block ────────────────────────────────────────────────────────
enc = mxt.mx_transformer_encoder_layer.from_encoder_layer(
    nn.TransformerEncoderLayer(512, 8), mxt.int8d, block=128)

# ── LoRA (QLoRA style) ───────────────────────────────────────────────────────
# Base weight frozen at int4d, trainable LoRA adapters in bfloat16
qlora = mxt.mx_lora_linear.from_linear(
    nn.Linear(512, 512), rank=16, base_dtype="int4d", lora_dtype=torch.bfloat16)
optimizer = torch.optim.AdamW(qlora.trainable_parameters())
merged = qlora.merge_weights()  # → mx_linear after training

# ── Sparse + quantized ────────────────────────────────────────────────────────
sparse_lin = mxt.mx_sparse_linear.from_linear(nn.Linear(512, 256),
             mxt.int4d, sparsity=0.5)

# ── Dynamic activation quantization ─────────────────────────────────────────
dyn = mxt.mx_dynamic_linear.from_linear(nn.Linear(512, 256),
      weight_dtype="int4d", act_dtype="int8d")
```

### Gradient / Training

mxtorch uses Straight-Through Estimator (STE) for gradients:

```python
# ── mx_quantize with STE ─────────────────────────────────────────────────────
x = torch.randn(256, requires_grad=True)
q = mxt.mx_quantize(x, "int8d", block=128)   # _MXQuantize.apply() under the hood
q.dequantize().sum().backward()
# x.grad ≠ None — STE passes gradient through quantize boundary

# ── _QGemmSTE: quantized GEMM with correct float-input STE ───────────────────
# Takes float tensors (not mx_tensor) for proper autograd tracking
from mxtorch import _QGemmSTE, get_mx_dtype
A = torch.randn(16, 32, requires_grad=True)
B = torch.randn(32, 16, requires_grad=True)
C = _QGemmSTE.apply(A, B, get_mx_dtype("int8d"), block=32)
C.sum().backward()
# A.grad and B.grad are populated — chain rule through quantized GEMM

# ── Training recipe ──────────────────────────────────────────────────────────
model = mxt.to_mx(base_model, "int8d")   # weight-only quant
opt   = mxt.mx_adam_w(model.parameters(), state_dtype="int8d")

for x, y in loader:
    out = model(x)                       # weight-only, fp32 compute
    if isinstance(out, mxt.mx_tensor):
        out = out.dequantize()
    loss = F.cross_entropy(out, y)
    loss.backward()                      # STE at all quant boundaries
    opt.step()
    opt.zero_grad()

# ── Stochastic rounding for training (zero-mean quantization noise) ─────────
q_train = mxt.stochastic_mx_quantize(grad, "int8ds", block=128)
```

### Optimizers

```python
# All states stored at MX precision (no fp32 momentum/variance buffers)
opt = mxt.mx_adam_w(model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    state_dtype="int8d",   # momentum and variance at int8
    block=128
)
```

### Model-Level Operations

```python
# ── Quantize entire model ────────────────────────────────────────────────────
model = mxt.to_mx(model, "int4d", block_size=128)
model = mxt.to_mx(model, {".*attn.*": "int8d", ".*mlp.*": "int4d"})
model = mxt.to_mx(model, "int8d", skip_patterns=["lm_head", "embed"])

# ── Inspect ──────────────────────────────────────────────────────────────────
print(mxt.inspect_model(model))

# ── Save / load ──────────────────────────────────────────────────────────────
# Only packed bits + scales — very small files
mxt.save_quantized(model, "model_q8.mx.pt")
model = mxt.load_quantized("model_q8.mx.pt", MyModelClass, dtype="int8d")

# ── Activation quantization ──────────────────────────────────────────────────
mxt.wrap_activations(model, "int8d")   # hooks on every layer output
mxt.unwrap_activations(model)

# ── Data-driven calibration ──────────────────────────────────────────────────
scales = mxt.calibrate(model, sample_batch, dtype="int8d",
                        percentile=99.9, n_samples=512)

# ── Context manager ──────────────────────────────────────────────────────────
with mxt.mx_mode("int4d", block=64):
    q = mxt.mx_quantize(x)   # uses int4d, block=64
```

### Analysis Tools

```python
snr_db = mxt.snr(weight, "int4d", block=128)
rmse   = mxt.quantization_error(weight, "int4d", metric="rmse")  # or "mae","max","relative"
table  = mxt.compare_dtypes(weight, ["int1d","int2d","int4d","int8d","float4d","float8d"])

# Precision audit: find any accidental fp32 tensors in the forward pass
with mxt.precision_audit(model) as audit:
    model(x)
print(audit.report())

# Hardware + roofline
hw = mxt.hardware_probe.detect()
print(mxt.hw_info())

est = mxt.roofline_estimator(hw)
res = est.estimate("matmul", mxt.int8d, (batch, seq, hidden), (hidden, hidden))
print(f"Bottleneck: {res.bottleneck}, compute util: {res.compute_util:.1%}")
```

### Distributed Training

#### DDP

```python
model = mxt.to_mx(model, "int8d")
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
mxt.install_ddp_hooks(model)   # gradient communication at MX precision
```

#### FSDP

mxtorch implements the PyTorch tensor subclass protocol for FSDP:

```python
# Verify protocol implementation
inner_names, meta = q.__tensor_flatten__()
# inner_names = ['_mx_packed', '_mx_scales']
# meta = {'mx_dtype': ..., 'orig_shape': ..., 'n': ..., 'block': ...}

q2 = mx_tensor.__tensor_unflatten__(
    {"_mx_packed": packed, "_mx_scales": scales}, meta, None, None)

# FSDP wrapping
policy = mxt.make_fsdp_mx_policy(min_params=1_000_000)
model  = mxt.mx_fsdp_wrapper.wrap(model, strategy="FULL_SHARD",
                                   device_id=local_rank, auto_wrap_policy=policy)

# Save / load with FSDP
mxt.mx_fsdp_wrapper.save_state_dict(model, "checkpoint.pt")
mxt.mx_fsdp_wrapper.load_state_dict(model, "checkpoint.pt")
```

> **FSDP2 (`torch.distributed._composable.fsdp.fully_shard`):** The tensor subclass protocol is implemented. Since `mx_tensor` IS a `torch.Tensor`, PyTorch's tree traversal treats it as a single tensor leaf (correct behavior for device placement). FSDP shards the packed storage. Requires PyTorch ≥ 2.3 for `fully_shard`.

> **NCCL:** No custom NCCL operations are required — mxtorch uses PyTorch's standard `dist.all_reduce` on the packed integer tensors, which works with any backend (NCCL, RCCL, Gloo).

### KV Cache Quantization

```python
cache = mxt.kv_cache_quantizer(n_heads=32, head_dim=128, dtype="int8d", max_len=8192)

# Per-token append (streaming inference)
for token in range(seq_len):
    k = compute_key(token)    # [1, n_heads, head_dim]
    v = compute_value(token)  # [1, n_heads, head_dim]
    cache.append_kv(k, v)

k_cache, v_cache = cache.get()   # full cached K and V
cache.reset()                     # clear for next sequence
```

### Custom Kernel Registration

```python
@mxt.register_kernel(
    op="torch.matmul",
    dtypes=["int4d"],
    hardware=["gfx1100"],   # ROCm gfx1100 (RX 7900 XTX)
    force="auto",           # "auto" | "true" | "false"
    priority=10
)
def my_optimised_int4_gemm():
    """Return Triton kernel source as string, or callable wrapper."""
    return my_kernel_fn   # called for all int4d matmuls on gfx1100

# Register for custom activation
@mxt.register_kernel(op="mx.gelu", dtypes=["int8d"], hardware=["sm_90"])
def my_fast_gelu_h100():
    return h100_gelu_kernel
```

---

## Performance

### Measured on AMD RX 7900 XTX (gfx1100, ROCm)

#### Quantize speed

| Dtype | 512×512 | 1024×1024 | notes |
|-------|---------|-----------|-------|
| int4d | 0.05 ms | 0.07 ms | bit packing |
| int8d | 0.05 ms | 0.06 ms | scale only |
| float4d | 0.24 ms | 0.30 ms | bucketize FP4 grid |
| float8d | 0.18 ms | 0.23 ms | bucketize FP8 grid |

#### GEMM throughput (int8d vs fp32)

| Shape | fp32 | int8d | speedup | GFLOPS |
|-------|------|-------|---------|--------|
| 256×256×256 | 0.033 ms | 0.073 ms | 0.5x | 91 |
| 512×512×512 | 0.086 ms | 0.085 ms | **1.0x** | 629 |
| 1024×1024×1024 | 0.275 ms | 0.190 ms | **1.5x** | 11,318 |

> GEMM break-even is around 512×512. At 1024×1024, int8d is 1.5× faster. This is the expected behavior: at small sizes, kernel launch dominates; at large sizes, bandwidth savings win.

#### Element-wise ops vs fp32

| Op | fp32 | int8d | float8d | int8d ratio |
|----|------|-------|---------|-------------|
| add (1M elem) | 0.014 ms | 0.249 ms | 0.528 ms | 0.06x |
| relu (1M elem) | 0.009 ms | 0.013 ms | 0.254 ms | **0.69x** |
| abs (1M elem) | 0.011 ms | 0.023 ms | 0.023 ms | **0.48x** |
| neg (1M elem) | 0.011 ms | 0.060 ms | 0.059 ms | 0.18x |

> relu (int8d) is near parity with fp32 via the packed `.clamp(min=0)` fast path. add is slower due to scale load + Q7 rescale overhead. This is inherent: fp32 add is one SASS instruction; int8 qadd is 6+ instructions. The win is bandwidth for large models (4× less data moved).

#### mx_linear vs nn.Linear (weight-only dequant inference)

| Shape | fp32 | int8d | int4d |
|-------|------|-------|-------|
| 512→512 (B=32) | 0.08 ms | 0.82 ms | 1.75 ms |
| 1024→512 | 0.13 ms | 0.83 ms | 1.89 ms |
| 2048→1024 | 0.14 ms | 0.82 ms | 2.16 ms |

> mx_linear is slower than fp32 because `from_linear` stores weights packed but the forward pass must dequantize them. This is the standard trade-off: weight memory is 4-8x smaller (good for VRAM), compute is slower (overhead from dequant). **Weight-only quantization wins on memory bandwidth-bound inference** (loading weights from VRAM), not on compute-bound inference.

#### Memory compression at 4096×4096

| Dtype | Total MB | vs fp32 | vs bf16 |
|-------|----------|---------|---------|
| fp32 | 67.1 | 1.0x | — |
| bfloat16 | 33.6 | 2.0x | 1.0x |
| int8d | 17.3 | **3.9x** | **1.9x** |
| int4d | 8.9 | **7.5x** | **3.8x** |
| int4d+DQ | 8.7 | **7.8x** | **3.9x** |

#### SNR Quality Table (N(0,1) weights)

| Dtype | Bits | SNR | RMSE |
|-------|------|-----|------|
| int1d | 1 | -6.5 dB | 2.16 |
| int2d | 2 | 2.3 dB | 0.78 |
| int4d | 4 | 18.7 dB | 0.118 |
| float4d | 4 | 19.3 dB | 0.108 |
| int4dh | 4 | 18.7 dB | 0.119 |
| float8d | 8 | 32.0 dB | 0.026 |
| int8d | 8 | **43.9 dB** | 0.007 |
| int8dh | 8 | 43.9 dB | 0.007 |
| float16d | 16 | ~96 dB | ~0.0 |

#### LSTM: mx_lstm int8d vs bfloat16 (RX 7900 XTX, hidden=512)

| Model | ms | GFLOPS | Weight MB | vs bf16 |
|-------|----|----|-------|---------|
| bfloat16 (baseline) | ~1.2 ms | ~87 | 8.4 | 1.0x |
| mx_lstm int8d | ~1.8 ms | ~59 | **2.1** | 0.7x |

> mx_lstm is slower due to weight dequantization per step. The primary benefit is **4× weight memory reduction** — for very large hidden sizes (4096+), VRAM reduction allows larger batch sizes.

---

## Troubleshooting

### "Triton not found" Warning

```
[mx_triton] Triton not found — pure-PyTorch fallback active.
```

Install Triton: `pip install triton`. On ROCm: `pip install triton-rocm`.

### Slow mx_linear (expected behavior)

mx_linear uses weight-only dequantization for inference. It's slower per-call than fp32 but uses 4-8× less VRAM. The intended use case is:
- Large models that don't fit in VRAM at fp32/bf16
- Batch inference where loading weights from VRAM is the bottleneck

For compute-bound inference with small batch sizes, consider using `int8d` or `int4d` only for the largest layers.

### float8d model quality degraded

```python
# float8d linear weight quality degrades because:
# - FP8 E4M3 max representable value is 448.0
# - For weight matrices with large outliers, the grid doesn't have enough points
# Recommended: use int8d for linear weights, float8d for activations
model = mxt.to_mx(model, {".*weight.*": "int8d"})  # int8d for weights
```

### Gradient not flowing

```python
# WRONG: _QGemmSTE takes floats, not mx_tensors
qc = _QGemmSTE.apply(qa_mxtensor, qb_mxtensor)  # ← TypeError

# CORRECT: pass float tensors, dtype and block size separately
qc = _QGemmSTE.apply(A_float, B_float, get_mx_dtype("int8d"), 128)
qc.sum().backward()  # A.grad and B.grad populated
```

### Out of Memory During float4d/float8d Quantize

Fixed in the current version: uses `torch.bucketize` instead of broadcasting.  
Peak extra VRAM is now ~6 MB for float4d at 512×512 (was 537 MB before).

### Debug Mode

```python
import os; os.environ["MX_DEBUG"] = "1"
import mxtorch as mxt
# Prints kernel dispatch decisions and fallback reasons
```

---

## API Classes Summary

| Class | Purpose |
|-------|---------|
| `mx_config` | Global settings (block_size, strict, debug) |
| `mx_dtype` | Quantization dtype descriptor |
| `mx_dtype_proxy` | torch.dtype-compatible wrapper |
| `mx_tensor` | Quantized tensor (torch.Tensor subclass) |
| `double_quantized` | Doubly-quantized tensor (scales also quantized) |
| `sparse_mx_tensor` | CSR sparse + MX quantized |
| `mx_linear` | Drop-in for nn.Linear |
| `mx_lstm` | Drop-in for nn.LSTM |
| `mx_gru` | Drop-in for nn.GRU |
| `mx_conv2d` | Drop-in for nn.Conv2d |
| `mx_multihead_attention` | Drop-in for nn.MultiheadAttention |
| `mx_layer_norm` / `mx_rms_norm` | Normalization with quantized affine |
| `mx_lora_linear` | QLoRA: frozen int4 base + trainable fp16 adapters |
| `mx_dynamic_linear` | Per-token dynamic activation quantization |
| `mx_adam_w` | AdamW with int8 optimizer states |
| `mx_quantizer` | Static class: all quantization methods |
| `mx_logical` | Static class: boolean ops |
| `mx_fused_ops` | Static class: fused kernels (SwiGLU, RoPE, SDPA) |
| `mx_analysis` | Static class: SNR, RMSE, compare_dtypes |
| `mx_model` | Static class: to_mx, save/load, calibrate |
| `mx_info` | Static class: hw_info, dtype_info, version |
| `mx_fsdp_wrapper` | FSDP integration (save/load state dict, auto_wrap_policy) |
| `kv_cache_quantizer` | Streaming KV cache at MX precision |

---

## License

GPL-3.0

## Citation

```bibtex
@software{mxtorch2026,
  title  = {mxtorch: Sub-byte MX Quantization for PyTorch with Triton GPU Kernels},
  author = {Daniel Derycke},
  year   = {2026},
  url    = {https://github.com/DHDev0/mxtorch/}
}
```
