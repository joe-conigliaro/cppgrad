// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#include <cmath>
#include <chrono>
#include <memory>
#include <random>
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "cppgrad/nn/mlp.h"
#include "cppgrad/nn/linear.h"
#include "cppgrad/nn/functional.h"
#include "cppgrad/ir/graph_context.h"
#include "cppgrad/utils/rng.h"
#include "cppgrad/optim/sgd.h"
#include "cppgrad/optim/adam.h"
#include "cppgrad/optim/adamw.h"
#include "cppgrad/backend/device_manager.h"
#include "examples/moons/moons_data.h"

using namespace cppgrad;

int main() {
    utils::global_rng().seed(42);
    backend::DeviceManager::instance().init();
    backend::DeviceManager::set_default_device(backend::DeviceType::CPU);
    // backend::DeviceManager::set_default_device(backend::DeviceType::METAL);
    // cppgrad::backend::cpu::Runtime::instance().set_num_threads(8);
    // cppgrad::backend::cpu::Runtime::instance().set_grain(128);


    std::cout << "--- CppGrad V2: Moons Full Training Loop ---" << std::endl;

    std::vector<float> X_data, y_data;
    MoonsParams P; P.n_samples=5000; P.noise=0.2f; P.seed=123;
    make_moons(P, X_data, y_data);

    auto xs = ir::from_vector(X_data, { (size_t)P.n_samples, 2 });
    auto ys = ir::from_vector(y_data, { (size_t)P.n_samples, 1 });

    // auto model = std::make_shared<nn::MLP>(2, 32, 1);
    nn::MLP model(2, 32, 1);
    auto params = model.parameters();

    // auto params = model->parameters();
    // for (size_t i = 0; i < params.size(); ++i) {
    //   auto& p = params[i];
    //   std::cerr << "[param] i="<<i
    //             << " leaf="<< p->is_leaf()
    //             << " req=" << p->requires_grad()
    //             << " node="<< p.get()
    //             << " data="<< (p->schedule() ? "y":"n")
    //             << "\n";
    // }
    // auto linear0 = std::dynamic_pointer_cast<nn::Linear>(model->direct_modules()[0]);
    // std::cerr << "[linear0] weight node="<< linear0->weight.get()
    //           << " bias node="<< (linear0->bias? linear0->bias.get():0) << "\n";


    // eager / no batching
    // optim::Adam optimizer(params, 0.05f);
    // lazy / batching
    optim::Adam optimizer(params, 0.17f);

    int epochs = 200;
    std::cout << "\n--- Starting Training ---" << std::endl;
    auto start = std::chrono::steady_clock::now();
    for (int e = 0; e < epochs; ++e) {
        ir::AutoGraphScope scope;
        // auto logits = (*model)(xs);
        auto logits = model(xs);
        // auto logits = model->forward(xs);

        auto loss = nn::functional::logistic_loss_pm1(logits, ys);

        optimizer.zero_grad();
        loss->backward();
        optimizer.step();

        if (e == 0 || (e + 1) % 10 == 0) {
            std::cout << "Epoch " << std::setw(3) << (e+1)
                      << " loss=" << std::fixed << std::setprecision(6) << loss->item<float>() << "\n";
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> secs = end - start;
    std::cout << "--- Training Complete (in " << std::fixed << std::setprecision(6) << secs.count() << " seconds) ---\n" << std::endl;


    std::cout << "Exporting original data + predictions to 'moons_original_preds.csv'..." << std::endl;
    std::ofstream original_file("examples/moons/moons_original_preds.csv");
    original_file << "x,y,actual,pred\n";

    auto final_preds = ir::tanh(model(xs));
    auto x_vec = xs->to_vector<float>();
    auto y_true_vec = ys->to_vector<float>();
    auto y_pred_vec = final_preds->to_vector<float>();

    for (int i = 0; i < P.n_samples; ++i) {
        original_file << x_vec[i * 2 + 0] << ","
                      << x_vec[i * 2 + 1] << ","
                      << y_true_vec[i] << ","
                      << y_pred_vec[i] << "\n";
    }
    original_file.close();
    std::cout << "Done." << std::endl;

    std::cout << "Generating and exporting decision boundary data to 'moons_boundary_preds.csv'..." << std::endl;
    std::vector<float> grid_points;
    int grid_size = 200;
    float x_min = -1.5, x_max = 2.5;
    float y_min = -1.0, y_max = 1.5;
    grid_points.reserve(grid_size * grid_size * 2);
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            grid_points.push_back(x_min + (x_max - x_min) * j / (grid_size - 1));
            grid_points.push_back(y_min + (y_max - y_min) * i / (grid_size - 1));
        }
    }

    auto xs_grid = ir::from_vector(grid_points, { (size_t)grid_size * grid_size, 2 });
    auto grid_preds = ir::tanh(model(xs_grid));
    auto grid_pred_vec = grid_preds->to_vector<float>();

    std::ofstream boundary_file("examples/moons/moons_boundary_preds.csv");
    boundary_file << "x,y,pred\n";
    for (int i = 0; i < grid_size * grid_size; ++i) {
        boundary_file << grid_points[i * 2 + 0] << ","
                      << grid_points[i * 2 + 1] << ","
                      << grid_pred_vec[i] << "\n";
    }
    boundary_file.close();
    std::cout << "Done." << std::endl;

    return 0;
}
