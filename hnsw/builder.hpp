#pragma once

#include "hnsw/graph.hpp"

namespace hnsw {

struct Builder {
  virtual void Build(float *data, int nb) = 0;
  virtual Graph<int> GetGraph() = 0;
  virtual ~Builder() = default;
};

} // namespace hnsw