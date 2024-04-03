#pragma once

#include <string>
#include <unordered_map>

#include "hnsw/quant/fp32_quant.hpp"
#include "hnsw/quant/sq4_quant.hpp"
#include "hnsw/quant/sq8_quant.hpp"

namespace hnsw {

enum class QuantizerType { FP32, SQ8, SQ4 };

inline std::unordered_map<int, QuantizerType> quantizer_map;

inline int quantizer_map_init = [] {
  quantizer_map[0] = QuantizerType::FP32;
  quantizer_map[1] = QuantizerType::SQ8;
  quantizer_map[2] = QuantizerType::SQ8;
  return 42;
}();

} // namespace hnsw
