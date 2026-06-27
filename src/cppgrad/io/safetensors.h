// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "cppgrad/ir/tensor.h"
#include "cppgrad/ir/tensor_utils.h"
#include "cppgrad/backend/device_manager.h"
#include "cppgrad/common/dtype.h"

namespace cppgrad::io {

// Convert BFloat16 (2 bytes) to float32.
static inline float bf16_to_float32(uint16_t b) {
    uint32_t f = static_cast<uint32_t>(b) << 16;
    return *reinterpret_cast<float*>(&f);
}

// Convert float32 to bfloat16 (round to nearest even).
static inline uint16_t float32_to_bf16(float v) {
    uint32_t x;
    std::memcpy(&x, &v, sizeof(x));
    if ((x & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((x >> 16) | 0x40u);  // keep NaN
    x += 0x7fffu + ((x >> 16) & 1u);  // round-to-nearest-even
    return (uint16_t)(x >> 16);
}

// Convert raw safetensors data to FP32 buffer, handling BF16/F16/F64/F32/INT32.
static std::vector<float> safetensors_to_float32(
    const std::vector<uint8_t>& raw,
    common::DType dtype,
    size_t numel)
{
    std::vector<float> result(numel);

    if (dtype == common::DType::BFLOAT16) {
        for (size_t i = 0; i < numel; ++i) {
            uint16_t b = static_cast<uint16_t>(raw[i * 2]) | (static_cast<uint16_t>(raw[i * 2 + 1]) << 8);
            result[i] = bf16_to_float32(b);
        }
    }
    else if (dtype == common::DType::FLOAT32) {
        std::copy_n(reinterpret_cast<const float*>(raw.data()), numel, result.data());
    }
    else if (dtype == common::DType::FLOAT64) {
        const double* d = reinterpret_cast<const double*>(raw.data());
        for (size_t i = 0; i < numel; ++i) result[i] = static_cast<float>(d[i]);
    }
    else if (dtype == common::DType::INT32) {
        const int32_t* d = reinterpret_cast<const int32_t*>(raw.data());
        for (size_t i = 0; i < numel; ++i) result[i] = static_cast<float>(d[i]);
    }
    else if (dtype == common::DType::FLOAT16) {
        for (size_t i = 0; i < numel; ++i) {
            uint16_t h = static_cast<uint16_t>(raw[i * 2]) | (static_cast<uint16_t>(raw[i * 2 + 1]) << 8);
            uint32_t sign = (h >> 15) & 0x1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f;
            if (exp == 0) {
                if (mant == 0) f = (sign << 31);
                else {
                    uint32_t m = mant << 1;
                    int s = 1;
                    while ((m & 0x400) == 0) { m <<= 1; s--; }
                    f = (sign << 31) | (static_cast<uint32_t>(std::max(0, 127 - 14 + s)) << 23) | ((m & 0x3FF) << 13);
                }
            } else if (exp == 31) {
                f = (sign << 31) | 0x7F800000 | (mant << 13);
            } else {
                f = (sign << 31) | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
            }
            result[i] = *reinterpret_cast<float*>(&f);
        }
    }
    else {
        throw std::runtime_error("safetensors: unsupported dtype for float32 conversion");
    }
    return result;
}

static common::DType safetensors_dtype(const std::string& dt) {
    if (dt == "F32")  return common::DType::FLOAT32;
    if (dt == "F64")  return common::DType::FLOAT64;
    if (dt == "F16")  return common::DType::FLOAT16;
    if (dt == "BF16") return common::DType::BFLOAT16;
    if (dt == "I32")  return common::DType::INT32;
    if (dt == "I64")  return common::DType::INT64;
    if (dt == "I8")   return common::DType::INT8;
    if (dt == "U8")   return common::DType::UINT8;
    if (dt == "U32")  return common::DType::UINT32;
    if (dt == "BOOL") return common::DType::BOOL8;
    throw std::runtime_error("safetensors: unsupported dtype '" + dt + "'");
}

static bool string_ends_with(const std::string& s, const std::string& suffix) {
    if (suffix.size() > s.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Read safetensors header, auto-detecting endianness (standard big-endian vs MLX little-endian).
// Returns {header_size, header_json_parsed}.
static std::pair<uint64_t, nlohmann::json> read_safetensors_header(std::ifstream& fs) {
    // Get file size for validation
    fs.seekg(0, std::ios::end);
    std::streamoff file_size = fs.tellg();
    fs.seekg(0);

    // Read 8-byte header length
    uint8_t buf8[8];
    fs.read(reinterpret_cast<char*>(buf8), 8);
    if (!fs) throw std::runtime_error("safetensors: failed to read header size");

    uint64_t header_size_be = 0, header_size_le = 0;
    for (int i = 0; i < 8; ++i) {
        header_size_be = (header_size_be << 8) | buf8[i];
        header_size_le = header_size_le | (static_cast<uint64_t>(buf8[i]) << (i * 8));
    }

    auto try_header_size = [&](uint64_t hs) -> bool {
        if (hs == 0 || hs > static_cast<uint64_t>(file_size) / 10) return false;
        fs.seekg(8);
        std::vector<char> test_buf(std::min(hs, static_cast<uint64_t>(100)));
        fs.read(test_buf.data(), static_cast<std::streamoff>(test_buf.size()));
        return test_buf[0] == '{';
    };

    uint64_t header_size = 0;
    if (try_header_size(header_size_be)) {
        header_size = header_size_be;
    } else if (try_header_size(header_size_le)) {
        header_size = header_size_le;
    } else {
        throw std::runtime_error("safetensors: could not determine header endianness");
    }

    // Read JSON header
    fs.seekg(8);
    std::vector<char> header_buf(static_cast<std::streamoff>(header_size));
    fs.read(header_buf.data(), static_cast<std::streamoff>(header_size));
    if (!fs) throw std::runtime_error("safetensors: failed to read header");

    // Parse the exact byte range -- the buffer is not null-terminated, and safetensors pads the
    // header with spaces (valid JSON whitespace), so parsing by length is required.
    return {header_size, nlohmann::json::parse(header_buf.begin(), header_buf.end())};
}

// Dequantize MLX 8-bit affine quantized weight to FP32 on CPU, then copy to device.
// weight: UINT32 tensor packed as int8 bytes (reinterpret U32 as bytes -> int8)
// scales: BF16/FP32 tensor [rows, num_groups]
// biases: BF16/FP32 tensor [rows, num_groups]
// group_size: typically 64
inline utils::Ref<ir::Tensor> dequant_mlx_affine(
    const utils::Ref<ir::Tensor>& weight_u32,
    const utils::Ref<ir::Tensor>& scales_raw,
    const utils::Ref<ir::Tensor>& biases_raw,
    backend::DeviceType device_type,
    int group_size = 64)
{
    auto device = backend::DeviceManager::device(device_type);
    if (!device) throw std::runtime_error("dequant: device not found");

    // Get weight data (U32 packed as int8 bytes)
    auto weight_buf = weight_u32->eval();
    if (!weight_buf) throw std::runtime_error("dequant: weight eval failed");

    // Get scales and biases as FP32 on CPU
    auto scales_buf_raw = scales_raw->eval();
    auto biases_buf_raw = biases_raw->eval();
    if (!scales_buf_raw || !biases_buf_raw) {
        throw std::runtime_error("dequant: scales/biases eval failed");
    }

    // Convert scales/biases to FP32 if needed. Read source buffers to host with a device-aware
    // copy (works for CPU and Metal); the inputs live on `device_type`, not necessarily CPU.
    std::vector<float> scales_f32, biases_f32;
    auto src_dev = backend::DeviceManager::device(device_type);

    auto convert_to_f32 = [&](const backend::Buffer& buf) -> std::vector<float> {
        std::vector<uint8_t> host(buf.size_bytes());
        src_dev->allocator()->copy_device_to_host(host.data(), buf);
        const uint8_t* data = host.data();
        std::vector<float> result(buf.numel());

        if (buf.dtype() == common::DType::BFLOAT16) {
            for (size_t i = 0; i < buf.numel(); ++i) {
                uint16_t b = data[i * 2] | (static_cast<uint16_t>(data[i * 2 + 1]) << 8);
                result[i] = bf16_to_float32(b);
            }
        } else if (buf.dtype() == common::DType::FLOAT32) {
            std::copy_n(reinterpret_cast<const float*>(data), buf.numel(), result.data());
        } else {
            throw std::runtime_error("dequant: unsupported scales dtype");
        }
        return result;
    };

    scales_f32 = convert_to_f32(*scales_buf_raw);
    biases_f32 = convert_to_f32(*biases_buf_raw);

    // Weight shape: [rows, packed_cols] where packed_cols = num_groups * group_size / 8
    // But actual packing: weight bytes = rows * dequant_cols (1:1 byte mapping)
    auto w_shape = weight_u32->shape();
    size_t rows = w_shape[0];
    size_t packed_cols = w_shape[1];

    // Total bytes per row = packed_cols * 4 (each U32 = 4 bytes)
    // These bytes represent int8 values: dequant_cols = packed_cols * 4
    size_t dequant_cols = packed_cols * 4;

    // scales shape: [rows, num_groups]
    auto s_shape = scales_raw->shape();
    size_t num_groups = s_shape[1];

    // Verify: dequant_cols should equal num_groups * group_size
    if (dequant_cols != num_groups * static_cast<size_t>(group_size)) {
        std::cerr << "[dequant] WARNING: dequant_cols=" << dequant_cols
                  << " != num_groups*group_size=" << (num_groups * group_size)
                  << " (using dequant_cols for output shape)\n";
    }

    // Read the quantized weight bytes to host (device-aware copy).
    std::vector<uint8_t> weight_host(weight_buf->size_bytes());
    src_dev->allocator()->copy_device_to_host(weight_host.data(), *weight_buf);
    const uint8_t* weight_bytes = weight_host.data();

    // Dequantize on CPU, storing the result as bfloat16 so large weight matrices stay at half
    // the memory of fp32 (e.g. ~54 GB vs ~108 GB for a 27B model). The matmul/gather paths
    // consume bf16 weights directly (fp32 activations + bf16 weights -> fp32 output).
    size_t actual_cols = num_groups * static_cast<size_t>(group_size);
    std::vector<uint16_t> dequant_bf16(rows * actual_cols, 0);

    // MLX affine quantization stores UNSIGNED integers (q in [0, 2^bits-1]); the dequant is
    // w = scale*q + bias, where `bias` (= w_min) absorbs the offset. Reading these bytes as
    // signed int8 corrupts every value >= 128 (flips it negative). Must be uint8.
    for (size_t r = 0; r < rows; ++r) {
        for (size_t g = 0; g < num_groups; ++g) {
            float scale = scales_f32[r * num_groups + g];
            float bias = biases_f32[r * num_groups + g];
            for (int k = 0; k < group_size; ++k) {
                size_t idx = r * num_groups * group_size + g * group_size + k;
                if (idx < static_cast<size_t>(rows * actual_cols)) {
                    uint8_t val = weight_bytes[idx];
                    dequant_bf16[idx] = float32_to_bf16(scale * static_cast<float>(val) + bias);
                }
            }
        }
    }

    auto out_buf = device->allocator()->allocate(rows * actual_cols, common::DType::BFLOAT16);
    device->allocator()->copy_host_to_device(*out_buf, dequant_bf16.data());

    return ir::Tensor::make_leaf(out_buf, {rows, actual_cols}, device_type, common::DType::BFLOAT16);
}

// Forward declaration for MLX dequantization (defined below).
inline std::map<std::string, utils::Ref<ir::Tensor>> dequantize_mlx_tensors(
    std::map<std::string, utils::Ref<ir::Tensor>> all_tensors,
    backend::DeviceType device_type);

// Load all tensors from safetensors files, with automatic BF16->FP32 conversion.
// For MLX quantized models (U32 weights + BF16 scales/biases): dequantizes to FP32.
// Supports both standard and MLX header formats.
inline std::map<std::string, utils::Ref<ir::Tensor>> load_safetensors(
    const std::string& path,
    backend::DeviceType device_type = backend::DeviceManager::default_device_type(),
    bool dequantize = true)
{
    std::ifstream fs(path, std::ios::binary);
    if (!fs.is_open()) {
        throw std::runtime_error("safetensors: cannot open file '" + path + "'");
    }

    auto [header_size, header] = read_safetensors_header(fs);

    auto* device = backend::DeviceManager::device(device_type);
    if (!device) throw std::runtime_error("safetensors: device not found");

    bool is_mlx = false;
    if (header.contains("__metadata__")) {
        auto& meta = header["__metadata__"];
        if (meta.contains("format") && meta["format"] == "mlx") {
            is_mlx = true;
        }
    }

    std::map<std::string, utils::Ref<ir::Tensor>> tensors;

    for (auto& [name, info] : header.items()) {
        if (name == "__metadata__") continue;

        std::string dtype_str = info["dtype"];
        auto& shape_arr = info["shape"];
        auto& offsets = info["data_offsets"];

        common::DType dtype = safetensors_dtype(dtype_str);

        std::vector<size_t> shape;
        for (auto& s : shape_arr) shape.push_back(static_cast<size_t>(s));

        size_t offset_begin = static_cast<size_t>(offsets[0]);
        size_t offset_end = static_cast<size_t>(offsets[1]);
        size_t data_size = offset_end - offset_begin;
        size_t numel = cppgrad::utils::vector::numel(shape);

        fs.seekg(8 + header_size + offset_begin);
        std::vector<uint8_t> raw(data_size);
        fs.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamoff>(data_size));
        if (!fs) throw std::runtime_error("safetensors: failed to read tensor '" + name + "'");

        // Convert BF16/F16/F64 to FP32; keep U32 for quantized weights
        if (dtype == common::DType::UINT32 && is_mlx) {
            // Quantized weight - store as U32 for later dequantization
            auto buffer = device->allocator()->allocate(numel, dtype);
            device->allocator()->copy_host_to_device(*buffer, raw.data());
            tensors[name] = ir::Tensor::make_leaf(buffer, shape, device_type, dtype);
        } else {
            auto f32_data = safetensors_to_float32(raw, dtype, numel);
            auto buffer = device->allocator()->allocate(numel, common::DType::FLOAT32);
            device->allocator()->copy_host_to_device(*buffer, f32_data.data());
            tensors[name] = ir::Tensor::make_leaf(buffer, shape, device_type, common::DType::FLOAT32);
        }
    }

    // If MLX format, dequantize U32 weights using scales and biases. Skipped when dequantize=false
    // (the multi-file loader defers this until all shards are merged, so quantized triples that
    // straddle a shard boundary are reunited before dequantization rather than dropped).
    if (is_mlx && dequantize) {
        return dequantize_mlx_tensors(tensors, device_type);
    }

    return tensors;
}

// Load tensors from multiple safetensors files and merge into a single map.
inline std::map<std::string, utils::Ref<ir::Tensor>> load_safetensors(
    const std::vector<std::string>& paths,
    backend::DeviceType device_type = backend::DeviceManager::default_device_type())
{
    std::map<std::string, utils::Ref<ir::Tensor>> all_tensors;

    for (auto& path : paths) {
        // Load each shard raw (no per-shard dequant) so quantized triples (weight/scales/biases)
        // split across shard boundaries are merged before dequantizing.
        auto tensors = load_safetensors(path, device_type, /*dequantize=*/false);
        for (auto& [name, tensor] : tensors) {
            all_tensors[name] = tensor;
        }
    }

    // Check if any U32 weights remain (quantized weights split across files)
    bool has_quantized = false;
    for (auto& [name, tensor] : all_tensors) {
        if (tensor->dtype() == common::DType::UINT32) {
            has_quantized = true;
            break;
        }
    }

    if (has_quantized) {
        return dequantize_mlx_tensors(all_tensors, device_type);
    }

    return all_tensors;
}

// Dequantize all MLX quantized tensors in a map, replacing U32 weights with FP32.
inline std::map<std::string, utils::Ref<ir::Tensor>> dequantize_mlx_tensors(
    std::map<std::string, utils::Ref<ir::Tensor>> all_tensors,
    backend::DeviceType device_type)
{
    std::map<std::string, utils::Ref<ir::Tensor>> result;

    // Find quantized weights and their corresponding scales/biases
    std::vector<std::string> quant_weight_names;
    for (auto& [name, tensor] : all_tensors) {
        if (tensor->dtype() == common::DType::UINT32 && string_ends_with(name, ".weight")) {
            quant_weight_names.push_back(name);
        }
    }

    // Dequantize each quantized weight
    for (auto& wname : quant_weight_names) {
        std::string base = wname.substr(0, wname.size() - 7); // remove ".weight"
        std::string sname = base + ".scales";
        std::string bname = base + ".biases";

        auto w_it = all_tensors.find(wname);
        auto s_it = all_tensors.find(sname);
        auto b_it = all_tensors.find(bname);

        if (w_it != all_tensors.end() && s_it != all_tensors.end() && b_it != all_tensors.end()) {
            if (w_it->second->dtype() == common::DType::UINT32) {
                auto dequantized = dequant_mlx_affine(w_it->second, s_it->second, b_it->second, device_type);
                result[wname] = dequantized;
                continue;
            }
        }
        // No scales/biases or already FP32 - keep as-is
        result[wname] = w_it->second;
    }

    // Copy non-quantized tensors (skip scales/biases that were used)
    for (auto& [name, tensor] : all_tensors) {
        if (result.find(name) != result.end()) continue;

        // Skip scale/bias tensors that were used for dequantization
        if (string_ends_with(name, ".scales") || string_ends_with(name, ".biases")) {
            // Check if there's a corresponding quantized weight
            std::string base = name.substr(0, name.size() - 7);
            std::string wname = base + ".weight";
            bool is_quant_base = false;
            for (auto& wname2 : quant_weight_names) {
                if (wname2 == wname) { is_quant_base = true; break; }
            }
            if (is_quant_base) continue; // skip scales/biases for quantized weights
        }

        result[name] = tensor;
    }

    return result;
}

} // namespace cppgrad::io
