// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <cmath>
#include <string>
#include <vector>
#include <cstddef>
#include <iostream>
#include <algorithm>

static int g_fail_count = 0;

#define EXPECT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "[FAIL] " << msg << " (at " << __FILE__ << ":" << __LINE__ << ")\n"; \
        g_fail_count++; \
    } \
} while(0)

#define EXPECT_CLOSE(val, exp, eps, msg) do { \
    if (std::fabs((val) - (exp)) > (eps)) { \
        std::cerr << "[FAIL] " << msg << " got=" << (val) << " expected=" << (exp) \
                  << " tol=" << (eps) << " (at " << __FILE__ << ":" << __LINE__ << ")\n"; \
        g_fail_count++; \
    } \
} while(0)

#define TEST_HEADER(name) std::cout << "\n=== " << name << " ===\n"

struct Tolerances {
    double atol = 1e-6;
    double rtol = 1e-5;
};

inline bool allclose(const float* a, const float* b, size_t n, const Tolerances& tol,
                     size_t* bad_idx = nullptr, float* got = nullptr, float* exp = nullptr, float* abs_err = nullptr) {
    for (size_t i = 0; i < n; ++i) {
        const float aa = a[i];
        const float bb = b[i];

        // const bool aa_nan = std::isnan(aa);
        // const bool bb_nan = std::isnan(bb);
        // // Treat NaN == NaN as equal for testing purposes
        // if (aa_nan || bb_nan) {
        //     if (aa_nan && bb_nan) continue; // both NaN -> OK
        //     if (bad_idx) *bad_idx = i;
        //     if (got) *got = aa;
        //     if (exp) *exp = bb;
        //     if (abs_err) *abs_err = std::numeric_limits<float>::quiet_NaN();
        //     return false;
        // }

        const double diff = std::fabs(double(aa) - double(bb));
        const double thr  = tol.atol + tol.rtol * std::fabs(double(bb));
        if (diff > thr) {
            if (bad_idx) *bad_idx = i;
            if (got) *got = aa;
            if (exp) *exp = bb;
            if (abs_err) *abs_err = float(diff);
            return false;
        }
    }
    return true;
}

inline void expect_allclose(const std::vector<float>& a, const std::vector<float>& b, float eps, const char* msg) {
    EXPECT_TRUE(a.size() == b.size(), std::string(msg) + " size mismatch");
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        EXPECT_CLOSE(a[i], b[i], eps, std::string(msg) + " idx=" + std::to_string(i));
    }
}

inline void print_mismatch(const std::string& tag, size_t idx, float got, float exp, float abs_err, const Tolerances& tol) {
    std::cerr << "[FAIL] " << tag << " idx=" << idx
              << " got=" << got << " expected=" << exp
              << " abs_err=" << (std::isnan(abs_err) ? std::string("nan") : std::to_string(abs_err))
              << " (atol=" << tol.atol << ", rtol=" << tol.rtol << ")\n";
    g_fail_count++;
}

// Convenience macro that uses allclose and reports the first mismatch
#define EXPECT_ALLCLOSE(ptr_a, ptr_b, n, tol, tag) do { \
    size_t _bad=0; float _got=0, _exp=0, _err=0; \
    if (!allclose((ptr_a), (ptr_b), (n), (tol), &_bad, &_got, &_exp, &_err)) { \
        print_mismatch((tag), _bad, _got, _exp, _err, (tol)); \
    } \
} while(0)

// Simple random filler (deterministic)
inline void fill_random(std::vector<float>& v, float lo=-1.0f, float hi=1.0f, unsigned seed=123) {
    uint32_t x = seed;
    auto rnd = [&]() {
        // simple LCG for deterministic runs
        x = 1664525u * x + 1013904223u;
        return (x & 0xFFFFFFu) / float(0xFFFFFFu);
    };
    for (auto& e : v) {
        float r = rnd();
        e = lo + (hi - lo) * r;
    }
}
