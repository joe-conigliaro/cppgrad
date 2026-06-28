// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// Batched matmul with broadcast batch dims (NumPy/torch semantics). Regression test for the bug where
// the builder computed the broadcast OUTPUT shape but passed operands with their original strides, so
// a size-1 (broadcast) batch dim was indexed with a nonzero stride -> silent wrong results. Covers
// size-1 dims on either operand, and rank mismatch (right-aligned). CPU + Metal.
#include <cmath>
#include <cstdio>
#include <vector>

#include "cppgrad/backend/device_manager.h"
#include "cppgrad/ir/grad_mode.h"
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/tensor_utils.h"
#include "tests/helpers.h"

using namespace cppgrad;
static uint32_t s = 11u;
static float rnd() {
    s = s * 1664525u + 1013904223u;
    return ((s >> 8) / float(1u << 24)) * 2.f - 1.f;
}
static std::vector<float> vec(size_t n) {
    std::vector<float> v(n);
    for (auto &x : v)
        x = rnd();
    return v;
}

// Reference batched matmul with full NumPy right-aligned batch broadcasting.
static std::vector<float> ref_bmm(const std::vector<float> &A, const std::vector<size_t> &as,
                                  const std::vector<float> &B, const std::vector<size_t> &bs) {
    size_t M = as[as.size() - 2], K = as[as.size() - 1], N = bs[bs.size() - 1];
    std::vector<size_t> ab(as.begin(), as.end() - 2), bb(bs.begin(), bs.end() - 2);
    size_t nb = std::max(ab.size(), bb.size()), pa = nb - ab.size(), pb = nb - bb.size();
    std::vector<size_t> ob(nb);
    size_t bc = 1;
    for (size_t i = 0; i < nb; ++i) {
        size_t ia = i < pa ? 1 : ab[i - pa], ib = i < pb ? 1 : bb[i - pb];
        ob[i] = std::max(ia, ib);
        bc *= ob[i];
    }
    std::vector<float> out(bc * M * N, 0.f);
    for (size_t bi = 0; bi < bc; ++bi) {
        // decompose bi over ob, map to a/b batch offsets (broadcast: size-1 -> idx 0)
        size_t aoff = 0, boff = 0, rem = bi, astr = 1, bstr = 1;
        std::vector<size_t> idx(nb);
        for (size_t d = nb; d-- > 0;) {
            idx[d] = rem % ob[d];
            rem /= ob[d];
        }
        for (size_t d = 0; d < ab.size(); ++d) {
            astr *= 1;
        }
        // compute strides
        size_t as_ = 1;
        for (size_t d = ab.size(); d-- > 0;) {
            size_t od = d + pa;
            aoff += (ab[d] == 1 ? 0 : idx[od]) * as_;
            as_ *= ab[d];
        }
        size_t bs_ = 1;
        for (size_t d = bb.size(); d-- > 0;) {
            size_t od = d + pb;
            boff += (bb[d] == 1 ? 0 : idx[od]) * bs_;
            bs_ *= bb[d];
        }
        const float *Ap = A.data() + aoff * M * K;
        const float *Bp = B.data() + boff * K * N;
        for (size_t m = 0; m < M; ++m)
            for (size_t n = 0; n < N; ++n) {
                float acc = 0;
                for (size_t k = 0; k < K; ++k)
                    acc += Ap[m * K + k] * Bp[k * N + n];
                out[(bi * M + m) * N + n] = acc;
            }
    }
    return out;
}

static bool one(backend::DeviceType dev, std::vector<size_t> as, std::vector<size_t> bs, const char *tag) {
    ir::NoGradScope ng;
    size_t an = 1;
    for (auto d : as)
        an *= d;
    size_t bn = 1;
    for (auto d : bs)
        bn *= d;
    auto A = vec(an), B = vec(bn);
    auto out = ir::matmul(ir::from_vector<float>(A, as, dev), ir::from_vector<float>(B, bs, dev))->to_vector<float>();
    auto ref = ref_bmm(A, as, B, bs);
    float w = 0;
    for (size_t i = 0; i < ref.size() && i < out.size(); ++i)
        w = std::max(w, std::fabs(ref[i] - out[i]));
    bool ok = out.size() == ref.size() && w < 1e-3f;
    std::printf("  [%s] %-22s : diff=%.2e %s\n", backend::to_string(dev), tag, w, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    backend::DeviceManager::instance().init();
    TEST_HEADER("batched matmul broadcast (right-aligned NumPy semantics)");
    std::vector<backend::DeviceType> devs = {backend::DeviceType::CPU};
    if (backend::DeviceManager::default_device_type() == backend::DeviceType::METAL)
        devs.push_back(backend::DeviceType::METAL);
    bool ok = true;
    for (auto d : devs) {
        ok &= one(d, {2, 3, 4, 5}, {2, 1, 5, 6}, "B size-1 dim");     // b broadcasts over dim1
        ok &= one(d, {2, 1, 4, 5}, {2, 3, 5, 6}, "A size-1 dim");     // a broadcasts over dim1
        ok &= one(d, {2, 3, 4, 5}, {2, 3, 5, 6}, "no broadcast");     // identity batch
        ok &= one(d, {3, 4, 5}, {5, 6}, "rank mismatch (b 2D)");      // b has no batch (right-align)
        ok &= one(d, {1, 3, 4, 5}, {2, 3, 5, 6}, "A leading size-1"); // a dim0=1 broadcasts to 2
        EXPECT_TRUE(ok, (std::string(backend::to_string(d)) + ": all broadcast cases").c_str());
    }
    if (g_fail_count == 0) {
        std::printf("\nALL TESTS PASSED (matmul broadcast)\n");
        return 0;
    }
    std::printf("\nFAILED (matmul broadcast)\n");
    return 1;
}
