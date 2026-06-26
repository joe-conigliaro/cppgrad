// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// gqa_attention (no-materialize grouped-query attention via stride-0 broadcast views) == the
// reference repeat_kv + scaled_dot_product_attention, to fp tolerance. Covers n_rep=1 (plain MHA),
// n_rep>1, S=1 (decode), S>1 (prefill), with and without a causal mask. CPU + Metal.
#include <cmath>
#include <vector>
#include <cstdint>
#include <cstdio>
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/ir/grad_mode.h"
#include "cppgrad/nn/functional.h"

using namespace cppgrad;

static uint32_t s = 7u;
static float rnd(){ s = s*1664525u + 1013904223u; return ((s>>8)/float(1u<<24))*2.f - 1.f; }
static utils::Ref<ir::Tensor> rt(const std::vector<size_t>& shp, backend::DeviceType d){
    size_t n=1; for(auto x:shp)n*=x; std::vector<float> v(n); for(auto&x:v)x=rnd();
    return ir::from_vector<float>(v,shp,d);
}
static utils::Ref<ir::Tensor> causal_mask(size_t S, size_t KV, backend::DeviceType d){
    std::vector<float> m(S*KV,0.f); size_t off=KV-S;
    for(size_t i=0;i<S;++i)for(size_t j=off+i+1;j<KV;++j) m[i*KV+j]=-1e9f;
    return ir::from_vector<float>(m,{1,1,S,KV},d);
}

static bool run(backend::DeviceType dev, size_t B, size_t S, size_t KV, size_t nKV, size_t n_rep, bool mask){
    ir::NoGradScope ng;
    size_t nH=nKV*n_rep, Dh=8;
    auto q = rt({B,S,nH,Dh}, dev);
    auto k = rt({B,KV,nKV,Dh}, dev);
    auto v = rt({B,KV,nKV,Dh}, dev);
    auto m = mask ? causal_mask(S,KV,dev) : nullptr;

    // reference: repeat_kv to nH heads then SDPA
    auto kr = (n_rep>1) ? nn::functional::repeat_kv(k,n_rep) : k;
    auto vr = (n_rep>1) ? nn::functional::repeat_kv(v,n_rep) : v;
    auto ref = nn::functional::scaled_dot_product_attention(q,kr,vr,m)->to_vector<float>();
    auto got = nn::functional::gqa_attention(q,k,v,m,n_rep)->to_vector<float>();

    float w=0; for(size_t i=0;i<ref.size();++i) w=std::max(w,std::fabs(ref[i]-got[i]));
    bool ok = w < 1e-3f;
    std::printf("  [%s] B=%zu S=%zu KV=%zu nKV=%zu n_rep=%zu mask=%d : diff=%.2e %s\n",
                backend::to_string(dev),B,S,KV,nKV,n_rep,(int)mask,w, ok?"OK":"FAIL");
    return ok;
}

int main(){
    backend::DeviceManager::instance().init();
    std::vector<backend::DeviceType> devs={backend::DeviceType::CPU};
    if (backend::DeviceManager::default_device_type()==backend::DeviceType::METAL) devs.push_back(backend::DeviceType::METAL);
    std::printf("=== gqa_attention == repeat_kv + SDPA ===\n");
    bool ok=true;
    for(auto d:devs){
        ok &= run(d, 1, 5, 5, 2, 1, true);    // MHA, prefill, causal
        ok &= run(d, 1, 5, 5, 2, 2, true);    // GQA n_rep=2, prefill, causal
        ok &= run(d, 1, 1, 9, 2, 4, false);   // decode (S=1), n_rep=4, no mask
        ok &= run(d, 1, 7, 12, 4, 2, true);   // prefill block at offset (KV>S), causal
        ok &= run(d, 2, 3, 3, 3, 2, true);    // B=2
        ok &= run(d, 1, 16, 16, 2, 3, true);  // larger
    }
    std::printf("\n%s\n", ok?"ALL TESTS PASSED (gqa attention)":"FAILED (gqa attention)");
    return ok?0:1;
}
