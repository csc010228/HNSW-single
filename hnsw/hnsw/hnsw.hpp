#pragma once

#include "hnsw/hnsw/HNSWInitializer.hpp"
#include "hnsw/builder.hpp"
#include "hnsw/common.hpp"
#include "hnsw/graph.hpp"
#include "hnsw/hnswlib/hnswlib.h"
#include "hnsw/hnswlib/space_ip.h"
#include "hnsw/hnswlib/space_l2.h"
#include "hnsw/hnswlib/hnswalg.h"
#include <chrono>
#include <memory>

namespace hnsw {

struct HNSW : public Builder {
  int nb, dim;
  int M, efConstruction;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw = nullptr;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space = nullptr;

  Graph<int> final_graph;

  HNSW(int dim, const std::string &metric, int R = 32, int L = 200)
      : dim(dim), M(R / 2), efConstruction(L) {
    auto m = metric_map[metric];
    if (m == Metric::L2) {
      space = std::make_unique<hnswlib::L2Space>(dim);
    } else if (m == Metric::IP) {
      space = std::make_unique<hnswlib::InnerProductSpace>(dim);
    } else {
      fprintf(stderr, "Unsupported metric type\n");
    }
  }

  void Build(float *data, int N) override {
    nb = N;
    hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), N, M,
                                                             efConstruction);
    int cnt = 0;
    auto st = std::chrono::high_resolution_clock::now();
    hnsw->addPoint(data, 0);
    for (int i = 1; i < nb; ++i) {
      hnsw->addPoint(data + i * dim, i);
      int cur = cnt += 1;
      if (cur % 10000 == 0) {
        fprintf(stderr, "HNSW building progress: [%d/%d]\n", cur, nb);
      }
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(ed - st).count();
    fprintf(stderr, "HNSW building cost: %.2lfs\n", ela);
    final_graph.init(nb, 2 * M);
    for (int i = 0; i < nb; ++i) {
      int *edges = (int *)hnsw->get_linklist0(i);
      for (int j = 1; j <= edges[0]; ++j) {
        final_graph.at(i, j - 1) = edges[j];
      }
    }
    auto initializer = std::make_unique<HNSWInitializer>(nb, M);
    initializer->ep = hnsw->enterpoint_node_;
    for (int i = 0; i < nb; ++i) {
      int level = hnsw->element_levels_[i];
      initializer->levels[i] = level;
      if (level > 0) {
        initializer->lists[i].assign(level * M, -1);
        for (int j = 1; j <= level; ++j) {
          int *edges = (int *)hnsw->get_linklist(i, j);
          for (int k = 1; k <= edges[0]; ++k) {
            initializer->at(j, i, k - 1) = edges[k];
          }
        }
      }
    }
    final_graph.initializer = std::move(initializer);
  }

  Graph<int> GetGraph() override { return final_graph; }
};
} // namespace hnsw