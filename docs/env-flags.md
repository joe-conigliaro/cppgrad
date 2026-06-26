# Environment flags

Runtime toggles read via `getenv`. Most are tuning/diagnostics; defaults are chosen so nothing needs
setting for normal use. (Build-time: `make DEBUG=true` defines `CPPGRAD_DEBUG` for debug builds.)

## Inference performance / memory tuning

| Flag | Default | Effect |
|------|---------|--------|
| `CPPGRAD_FLASH_ATTN=1` | off | Use fused flash attention (online softmax, no `[S,KV]` score matrix → O(1) attention memory) instead of the default `gqa_attention`. Opt-in while being validated; bit-equivalent. |
| `CPPGRAD_KV_CAPACITY=N` | on-demand | Preallocate the persistent prefix-cache KV cache to `N` positions so cross-request reuse survives a growing conversation. Set to your max context (e.g. `65536`) for Claude-Code-style fixed prefixes; otherwise the cache is sized per request and reuse is lost when the conversation outgrows it. |
| `CPPGRAD_PREFILL_CHUNK=N` | 256 | Prefill chunk size (tokens per forward). Lower it if a very long prompt still exhausts GPU memory. |
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
