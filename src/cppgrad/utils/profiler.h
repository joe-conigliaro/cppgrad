// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

// Lightweight, reusable profiler. Enabled by the env var CPPGRAD_PROFILE (zero cost when off).
//
// Two things are recorded, both keyed by a free-form category string:
//   - memory traffic (bytes moved in+out) and op count, accumulated per category. In a
//     memory-bandwidth-bound regime (e.g. quantized LLM decode) bytes are the best proxy for
//     time, so the by-bytes breakdown tells you which kernels actually cost.
//   - wall/GPU time (nanoseconds), for regions you explicitly time (ProfileScope) or backend
//     hooks that have a real duration (e.g. a Metal command buffer's GPU time).
//
// Usage:
//   if (Profiler::enabled()) Profiler::instance().record("MatMulOp", /*ns=*/0, bytes);
//   { ProfileScope _("decode"); ... }                 // times the scope, attributes to "decode"
//   Profiler::instance().report();                    // prints a sorted table to stderr
//   Profiler::instance().reset();                     // clear between phases (e.g. after prefill)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

namespace cppgrad::utils {

class Profiler {
public:
    struct Stat { uint64_t count = 0; uint64_t bytes = 0; double ns = 0.0; };

    static Profiler& instance() { static Profiler p; return p; }

    // Cached once: the profiler is a dev tool, toggled per-process via the environment.
    static bool enabled() {
        static const bool e = std::getenv("CPPGRAD_PROFILE") != nullptr;
        return e;
    }

    void record(const std::string& category, double ns, uint64_t bytes, uint64_t count = 1) {
        auto& s = _stats[category];
        s.count += count;
        s.bytes += bytes;
        s.ns    += ns;
    }

    void reset() { _stats.clear(); }

    void report(std::FILE* out = stderr, const char* title = "profile") const {
        if (_stats.empty()) return;
        std::vector<std::pair<std::string, Stat>> v(_stats.begin(), _stats.end());
        // Sort by memory traffic (the bandwidth-bound cost proxy), then by time.
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
            if (a.second.bytes != b.second.bytes) return a.second.bytes > b.second.bytes;
            return a.second.ns > b.second.ns;
        });
        uint64_t tb = 0, tc = 0; double tns = 0.0;
        for (const auto& p : v) { tb += p.second.bytes; tc += p.second.count; tns += p.second.ns; }
        std::fprintf(out, "\n[%s] %llu ops, %.1f MB traffic, %.2f ms timed\n", title,
                     (unsigned long long)tc, tb / 1e6, tns / 1e6);
        std::fprintf(out, "  %-24s %9s %12s %10s %7s\n", "category", "count", "MB", "ms", "%MB");
        for (const auto& p : v) {
            std::fprintf(out, "  %-24s %9llu %12.2f %10.3f %6.1f%%\n",
                         p.first.c_str(), (unsigned long long)p.second.count,
                         p.second.bytes / 1e6, p.second.ns / 1e6,
                         tb ? 100.0 * p.second.bytes / tb : 0.0);
        }
    }

private:
    std::map<std::string, Stat> _stats;
};

// RAII timer for a named region. No-op (no allocation, no clock read attributed) when disabled.
class ProfileScope {
public:
    explicit ProfileScope(std::string category, uint64_t bytes = 0)
        : _category(std::move(category)), _bytes(bytes), _on(Profiler::enabled()) {
        if (_on) _t0 = std::chrono::steady_clock::now();
    }
    ~ProfileScope() {
        if (!_on) return;
        const double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - _t0).count();
        Profiler::instance().record(_category, ns, _bytes);
    }
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;

private:
    std::string _category;
    uint64_t _bytes;
    bool _on;
    std::chrono::steady_clock::time_point _t0;
};

} // namespace cppgrad::utils
