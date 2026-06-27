# Environment flags

Runtime toggles read via `getenv`. Most are tuning/diagnostics; defaults are chosen so nothing needs
setting for normal use. (Build-time: `make DEBUG=true` defines `CPPGRAD_DEBUG` for debug builds.)

## Inference performance / memory tuning

| Flag | Default | Effect |
|------|---------|--------|
| `CPPGRAD_FLASH_ATTN=1` | off | Use fused flash attention (online softmax, no `[S,KV]` score matrix → O(1) attention memory) instead of the default `gqa_attention`. Opt-in while being validated; bit-equivalent. |
| `CPPGRAD_KV_CAPACITY=N` | on-demand | Preallocate the persistent prefix-cache KV cache to `N` positions so cross-request reuse survives a growing conversation. Set to your max context (e.g. `65536`) for Claude-Code-style fixed prefixes; otherwise the cache is sized per request and reuse is lost when the conversation outgrows it. |
| `CPPGRAD_KV_CACHE_FILE=/path` | off | Persist the warm prompt-prefix KV cache to disk: loaded at server startup, saved once after the first request. A fixed system+tools prompt is then prefilled once on a machine, ever - a server restart reloads the cache instead of re-prefilling (the file is large: ~the KV cache size; rejected if the model config differs). Halved by `CPPGRAD_KV_DTYPE=bf16`. |
| `CPPGRAD_KV_DTYPE=bf16` | `f32` | Store the full-attention KV cache in bfloat16 instead of fp32: halves cache memory + read bandwidth at long context (a decode win that grows with prefix length) and halves the persisted cache file. Attention accumulates in fp32, so accuracy loss is bf16-rounding only. fp32 activations are converted on the cache write; bf16 attention compute is Metal-only. |
| `CPPGRAD_BF16_ACT=1` | off | Run the FFN intermediate in bf16 (gate/up GEMMs emit bf16, silu+mul run bf16, down GEMM consumes bf16 → fp32 residual). Halves the FFN's elementwise + GEMM-I/O traffic; fp32-accumulate so loss is bf16-rounding only. Prefill only (M>1 tiled GEMM; decode's GEMV stays fp32), quantized SwiGLU only. Opt-in while being validated on real models. |
| `CPPGRAD_PREFILL_CHUNK=N` | 256 | Prefill chunk size (tokens per forward). Bigger = higher prefill throughput (better GPU occupancy, fewer launches), but a bigger per-chunk command buffer - pair with `CPPGRAD_METAL_MAX_KERNELS` to avoid OOM. Lower it if a very long prompt still exhausts GPU memory. |
| `CPPGRAD_METAL_MAX_KERNELS=N` | 0 (off) | Flush the Metal command buffer after every `N` kernels instead of only at scope/chunk boundaries. Bounds resident GPU memory independent of chunk size, so large `CPPGRAD_PREFILL_CHUNK` values don't OOM (a big chunk's linear-attn scan can emit tens of thousands of kernels). Order-preserving (flush = commit+wait); lower `N` = less memory, slightly more sync overhead. Try `8192` with `CPPGRAD_PREFILL_CHUNK=512`. |
| `CPPGRAD_PREFILL_AREA=N` | 8000000 | Adaptive-chunk budget: caps the chunk so `chunk*(offset+chunk) ≤ N`, bounding the attention score-matrix transient as the prefix grows. `0` disables the cap. (Moot under `CPPGRAD_FLASH_ATTN`, which never materializes scores.) |
| `CPPGRAD_DELTA_CHUNK=N` | 32 | Linear-attention (GatedDeltaNet) scan sub-chunk size. Larger = fewer/bigger matmuls (faster) but the de-decay step is more numerically demanding; lower if a strong-decay model produces garbage. |

## Opt-outs / fallbacks

| Flag | Effect |
|------|--------|
| `CPPGRAD_NO_PREFIX_CACHE=1` | Disable cross-request prefix KV-cache reuse (plain and speculative paths). |
| `QWEN_KV_CONCAT=1` | Use the concat-mode KV cache (the O(n²) reference path) instead of the in-place cache. |

## Diagnostics / profiling

| Flag | Effect |
|------|--------|
| `CPPGRAD_PREFILL_QUIET=1` | Silence the per-chunk `[prefill] … tok/s … ETA` progress log. |
| `CPPGRAD_PROFILE=1` | Per-op memory-traffic + GPU-time profiler report. |
| `CPPGRAD_METAL_DISPATCH=1` | Log the kernel count of each Metal flush. |
| `CPPGRAD_METAL_CAPTURE=N` | Capture the Nth Metal flush to `/tmp/cppgrad_flush.gputrace` (run the process under `METAL_CAPTURE_ENABLED=1`). |
| `QWEN_TIMING=1` | Print prefill time and decode tokens/sec. |
| `QWEN_DEBUG=1` | Per-layer hidden-state magnitudes on the first decode step. |
| `QWEN_SPEC_DEBUG=1` | Speculative decode: compare incremental-cache verify against a fresh-cache verify (catches cache aliasing). |
