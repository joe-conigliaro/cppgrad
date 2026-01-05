// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <vector>
#include <random>
#include <cmath>

struct MoonsParams {
    int n_samples = 1000; // total points
    float radius = 1.0f;
    float dx = 1.0f;      // inner moon x offset
    float dy = 0.0f;      // inner moon y offset
    float noise = 0.2f;   // Gaussian noise stddev
    uint32_t seed = 42;
};

// Generates two moons:
inline void make_moons(const MoonsParams& P, std::vector<float>& X, std::vector<float>& y) {
    if (P.n_samples <= 0) { X.clear(); y.clear(); return; }
    if (P.radius <= 0.0f) throw std::runtime_error("make_moons_reflect: radius must be > 0");

    X.clear(); y.clear();
    X.reserve(static_cast<size_t>(P.n_samples) * 2);
    y.reserve(static_cast<size_t>(P.n_samples));

    std::mt19937 gen(P.seed);
    std::uniform_real_distribution<float> Uang(0.0f, static_cast<float>(M_PI));
    const bool add_noise = (P.noise > 0.0f);
    std::normal_distribution<float> N(0.0f, P.noise);

    const int half = P.n_samples / 2;

    // Upper moon (+1): (r cos θ, r sin θ)
    for (int i = 0; i < half; ++i) {
        const float th = Uang(gen);
        float x  = P.radius * std::cos(th);
        float yy = P.radius * std::sin(th);
        if (add_noise) { x += N(gen); yy += N(gen); }
        X.push_back(x); X.push_back(yy);
        y.push_back(1.0f);
    }

    // Lower moon (−1): reflect y, then translate (dx, dy)
    for (int i = 0; i < P.n_samples - half; ++i) {
        const float th = Uang(gen);
        float x  = P.radius * std::cos(th) + P.dx;
        float yy = -P.radius * std::sin(th) + P.dy;
        if (add_noise) { x += N(gen); yy += N(gen); }
        X.push_back(x); X.push_back(yy);
        y.push_back(-1.0f);
    }
}
