// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <memory>
#include <vector>
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_ops.h"
#include "cppgrad/ir/graph_context.h"
#include "cppgrad/optim/optim.h"

namespace cppgrad {
namespace optim {

class Adam : public Optimizer {
public:
    Adam(std::vector<utils::Ref<ir::Tensor>> params, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f)
    : Optimizer(std::move(params), lr), _beta1(beta1), _beta2(beta2), _eps(eps), _weight_decay(weight_decay), _t(0) {
        init_state();
    }

    void step() override {
        ++_t;
        const float b1t = std::pow(_beta1, _t);
        const float b2t = std::pow(_beta2, _t);

        for (size_t i = 0; i < _params.size(); ++i) {
            auto& p = _params[i];
            if (!p) continue;

            auto g = p->grad();
            if (!g) continue;

            auto g_eff = apply_weight_decay_to_grad(g, p, _weight_decay);

            auto m_new = (_m[i] * _beta1) + (g_eff * (1.0f - _beta1));
            auto v_new = (_v[i] * _beta2) + (g_eff * g_eff * (1.0f - _beta2));

            auto m_hat = m_new / (1.0f - b1t);
            auto v_hat = v_new / (1.0f - b2t);

            auto step_core = (_lr * m_hat) / ir::sqrt(v_hat + _eps);
            auto p_new = update_parameters(p, step_core, _weight_decay);

            _m[i]->assign(m_new)->schedule();
            _v[i]->assign(v_new)->schedule();
            p->assign(p_new)->schedule();
        }
    }

protected:
    void init_state() {
        _m.clear(); _v.clear();
        _m.reserve(_params.size()); _v.reserve(_params.size());
        for (auto& p : _params) {
            if (!p) { _m.push_back(nullptr); _v.push_back(nullptr); continue; }
            auto m = ir::parameterize(ir::zeros_like(p)); m->set_requires_grad(false);
            auto v = ir::parameterize(ir::zeros_like(p)); v->set_requires_grad(false);
            _m.push_back(m);
            _v.push_back(v);
        }
    }

    virtual utils::Ref<ir::Tensor>
    apply_weight_decay_to_grad(const utils::Ref<ir::Tensor>& g, const utils::Ref<ir::Tensor>& p, float wd) {
        if (wd == 0.0f) return g;
        return g + (p * wd); // coupled L2 regularization
    }

    virtual utils::Ref<ir::Tensor>
    update_parameters(const utils::Ref<ir::Tensor>& p, const utils::Ref<ir::Tensor>& step_core, float /*wd*/) {
        return p - step_core;
    }

private:
    float _beta1, _beta2, _eps;
    float _weight_decay = 0.0f;
    int   _t;
    std::vector<utils::Ref<ir::Tensor>> _m, _v;
};

} // namespace optim
} // namespace cppgrad
