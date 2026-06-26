// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// bf16 KV cache. Two properties:
//   (1) The converting cache_update write (fp32 activations -> bf16 cache) round-trips to exactly the
//       host RNE bf16 rounding -- on CPU and Metal (the copy_view f32->bf16 kernel).
//   (2) Attention over a bf16 cache (flash_attention_bf16kv / gqa with bf16 K/V, fp32 accumulate)
//       matches attention over an fp32 cache to bf16 tolerance -- Metal (bf16 compute is GPU-only).
#include <cmath>
#include <vector>
#include <cstdint>
#include <cstdio>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/common/bfloat16.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/parameter.h"
#include "cppgrad/ir/grad_mode.h"
#include "cppgrad/nn/functional.h"

using namespace cppgrad;
static uint32_t s = 11u;
static float rnd(){ s=s*1664525u+1013904223u; return ((s>>8)/float(1u<<24))*2.f-1.f; }
static utils::Ref<ir::Tensor> rt(const std::vector<size_t>& shp, backend::DeviceType d){
    size_t n=1; for(auto x:shp)n*=x; std::vector<float> v(n); for(auto&x:v)x=rnd();
    return ir::from_vector<float>(v,shp,d);
}

// (1) cache_update fp32 -> bf16 leaf == host RNE bf16 rounding.
static bool roundtrip(backend::DeviceType dev){
    ir::NoGradScope ng;
    const size_t KV=37, nKV=4, Dh=16;
    auto k = rt({1,KV,nKV,Dh}, dev);
    auto host = k->to_vector<float>();
    auto cache = ir::parameter({1,KV,nKV,Dh}, dev, common::DType::BFLOAT16, true);
    cache->set_requires_grad(false);
    auto wrote = ir::cache_update(cache, k, /*axis=*/1, /*start=*/0);   // converting write
    wrote->eval();
    auto back = cache->to_vector<common::bfloat16>();
    float w=0; for(size_t i=0;i<host.size();++i){
        float expect = (float)common::bfloat16(host[i]);   // host RNE rounding
        w = std::max(w, std::fabs(expect - (float)back[i]));
    }
    bool ok = (w==0.f);
    std::printf("  [%s] cache_update f32->bf16 roundtrip : max|diff|=%.2e %s\n", backend::to_string(dev), w, ok?"OK":"FAIL");
    return ok;
}

// (2) attention over bf16 cache vs fp32 cache (Metal).
static bool attn(backend::DeviceType dev, size_t S, size_t KV, size_t nKV, size_t n_rep, size_t Dh){
    ir::NoGradScope ng;
    size_t nH=nKV*n_rep, off=KV-S;
    auto q = rt({1,S,nH,Dh}, dev);
    auto k = rt({1,KV,nKV,Dh}, dev);
    auto v = rt({1,KV,nKV,Dh}, dev);

    auto fill = [&](common::DType dt, const utils::Ref<ir::Tensor>& src){
        auto c = ir::parameter({1,KV,nKV,Dh}, dev, dt, true); c->set_requires_grad(false);
        return ir::cache_update(c, src, 1, 0);   // [1,KV,nKV,Dh] read view of the (possibly bf16) cache
    };
    auto kf32 = fill(common::DType::FLOAT32, k), vf32 = fill(common::DType::FLOAT32, v);
    auto kbf  = fill(common::DType::BFLOAT16, k), vbf  = fill(common::DType::BFLOAT16, v);

    auto ref_flash = nn::functional::flash_attention(q,kf32,vf32,n_rep,true,off)->to_vector<float>();
    auto got_flash = nn::functional::flash_attention(q,kbf, vbf, n_rep,true,off)->to_vector<float>();
    auto ref_gqa   = nn::functional::gqa_attention(q,kf32,vf32,nullptr,n_rep)->to_vector<float>();  // non-causal ok: same cache
    auto got_gqa   = nn::functional::gqa_attention(q,kbf, vbf, nullptr,n_rep)->to_vector<float>();

    float wf=0,wg=0;
    for(size_t i=0;i<ref_flash.size();++i) wf=std::max(wf,std::fabs(ref_flash[i]-got_flash[i]));
    for(size_t i=0;i<ref_gqa.size();++i)   wg=std::max(wg,std::fabs(ref_gqa[i]-got_gqa[i]));
    bool ok = wf<3e-2f && wg<3e-2f;
    std::printf("  [%s] S=%zu KV=%zu nKV=%zu n_rep=%zu Dh=%zu : flash|diff|=%.2e gqa|diff|=%.2e %s\n",
                backend::to_string(dev),S,KV,nKV,n_rep,Dh,wf,wg, ok?"OK":"FAIL");
    return ok;
}

int main(){
    backend::DeviceManager::instance().init();
    bool metal = backend::DeviceManager::default_device_type()==backend::DeviceType::METAL;
    std::printf("=== bf16 KV cache ===\n");
    bool ok=true;

    // (1) converting write round-trip: CPU + Metal.
    ok &= roundtrip(backend::DeviceType::CPU);
    if (metal) ok &= roundtrip(backend::DeviceType::METAL);

    // (2) bf16-cache attention == fp32-cache attention (bf16 compute is Metal-only).
    if (metal) {
        ok &= attn(backend::DeviceType::METAL, 5, 5, 2, 1, 128);
        ok &= attn(backend::DeviceType::METAL, 1, 40, 4, 2, 256);   // decode, Qwen3.6 head_dim
        ok &= attn(backend::DeviceType::METAL, 8, 32, 4, 2, 256);   // prefill block
    } else {
        std::printf("  [skip] Metal unavailable -- bf16 attention compute test skipped\n");
    }

    std::printf("\n%s\n", ok?"ALL TESTS PASSED (bf16 KV cache)":"FAILED (bf16 KV cache)");
    return ok?0:1;
}
