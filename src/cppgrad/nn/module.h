// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/parameter.h"

namespace cppgrad {
namespace nn {

class Module : public std::enable_shared_from_this<Module> {
public:
    virtual ~Module() = default;

    utils::Ref<ir::Tensor> operator()(const utils::Ref<ir::Tensor>& input) {
        return this->forward(input);
    }

    virtual utils::Ref<ir::Tensor> forward(const utils::Ref<ir::Tensor>& input) {
        auto current_x = input;
        for (const auto& module : _modules_vec) {
            current_x = (*module)(current_x);
        }
        return current_x;
    }

    std::vector<utils::Ref<ir::Tensor>> parameters() {
        std::vector<utils::Ref<ir::Tensor>> params;
        params.reserve(_parameters.size());
        for (auto const& [name, param] : _parameters) params.push_back(param);
        for (auto const& module : _modules_vec) {
            auto sub_params = module->parameters();
            params.insert(params.end(), sub_params.begin(), sub_params.end());
        }
        return params;
    }

    std::vector<utils::Ref<ir::Tensor>> direct_parameters() {
        std::vector<utils::Ref<ir::Tensor>> params;
        params.reserve(_parameters.size());
        for (auto const& [name, param] : _parameters) params.push_back(param);
        return params;
    }

    std::vector<std::shared_ptr<Module>> direct_modules() {
        return _modules_vec;
    }

    // utils::Ref<ir::Tensor> get_parameter(const std::string& name) const {
    //     auto it = _parameters.find(name);
    //     if (it == _parameters.end()) throw std::runtime_error("get_parameter: not found: " + name);
    //     return it->second;
    // }

    // // Replace parameter handle (used by optimizer or model to refresh pointers)
    // void set_parameter(const std::string& name, utils::Ref<ir::Tensor> param) {
    //     if (!param) throw std::runtime_error("set_parameter: null tensor");
    //     auto it = _parameters.find(name);
    //     if (it == _parameters.end()) throw std::runtime_error("set_parameter: not found: " + name);
    //     // ensure canonical and flags
    //     if (!param->is_canonical_leaf()) throw std::runtime_error("set_parameter: non-canonical leaf");
    //     param->set_requires_grad(true); // keep trainable
    //     _parameters[name] = std::move(param);
    // }

protected:
    void register_parameter(const std::string& name, utils::Ref<ir::Tensor> param) {
        if (!param) throw std::runtime_error("register_parameter: null tensor");
        // if (!param->is_leaf()) throw std::runtime_error("register_parameter: param must be a leaf");
        if (!param->is_canonical_leaf()) throw std::runtime_error("register_parameter: non-canonical leaf");
        if (_parameters.count(name)) throw std::runtime_error("register_parameter: parameter already exists: " + name);

        param->schedule(); // ensure storage exists
        param->set_requires_grad(true);
        _parameters[name] = std::move(param);
    }


    void register_module(const std::string& name, std::shared_ptr<Module> module) {
        _named_modules[name] = module;
        _modules_vec.push_back(module);
    }

    void register_modules(std::vector<std::shared_ptr<Module>> modules) {
        _modules_vec = std::move(modules);
    }

protected:
    std::vector<std::shared_ptr<Module>> _modules_vec;

private:
    std::map<std::string, utils::Ref<ir::Tensor>> _parameters;
    std::map<std::string, std::shared_ptr<Module>> _named_modules;
};

} // namespace nn
} // namespace cppgrad
