/*
HNSW
======================================================================================================

    HNSW算法
    参考: Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs

======================================================================================================
得分：
*/


/* =================================================================================================== */
/*                                             common.hpp                                              */
#include <string>
#include <unordered_map>

#define DEBUG_ENABLED // 定义调试输出开关

namespace hnsw {

enum class Metric {
  L2,
  IP,
};

inline std::unordered_map<std::string, Metric> metric_map;

inline constexpr size_t upper_div(size_t x, size_t y) {
  return (x + y - 1) / y;
}

inline constexpr int64_t do_align(int64_t x, int64_t align) {
  return (x + align - 1) / align * align;
}

#if defined(__clang__)

#define FAST_BEGIN
#define FAST_END
#define GLASS_INLINE __attribute__((always_inline))

#elif defined(__GNUC__)

#define FAST_BEGIN                                                             \
  _Pragma("GCC push_options") _Pragma(                                         \
      "GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define FAST_END _Pragma("GCC pop_options")
#define GLASS_INLINE [[gnu::always_inline]]
#else

#define FAST_BEGIN
#define FAST_END
#define GLASS_INLINE

#endif

} // namespace hnsw
/* =================================================================================================== */
/*                                             memory.hpp                                              */
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

namespace hnsw {

template <typename T> struct align_alloc {
  T *ptr = nullptr;
  using value_type = T;
  T *allocate(int n) {
    if (n <= 1 << 14) {
      int sz = (n * sizeof(T) + 63) >> 6 << 6;
      return ptr = (T *)aligned_alloc(64, sz);
    }
    int sz = (n * sizeof(T) + (1 << 21) - 1) >> 21 << 21;
    ptr = (T *)aligned_alloc(1 << 21, sz);
    return ptr;
  }
  void deallocate(T *, int) { free(ptr); }
  template <typename U> struct rebind {
    typedef align_alloc<U> other;
  };
  bool operator!=(const align_alloc &rhs) { return ptr != rhs.ptr; }
};

inline void *alloc2M(size_t nbytes) {
  size_t len = (nbytes + (1 << 21) - 1) >> 21 << 21;
//   auto p = std::aligned_alloc(1 << 21, len);
  auto p = aligned_alloc(1 << 21, len);
  std::memset(p, 0, len);
  return p;
}

inline void *alloc64B(size_t nbytes) {
  size_t len = (nbytes + (1 << 6) - 1) >> 6 << 6;
//   auto p = std::aligned_alloc(1 << 6, len);
  auto p = aligned_alloc(1 << 6, len);
  std::memset(p, 0, len);
  return p;
}

} // namespace hnsw
/* =================================================================================================== */
/*                                            neighbor.hpp                                             */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <vector>

namespace hnsw {

namespace searcher {

template <typename Block = uint64_t> struct Bitset {
  constexpr static int block_size = sizeof(Block) * 8;
  int nbytes;
  Block *data;
  explicit Bitset(int n)
      : nbytes((n + block_size - 1) / block_size * sizeof(Block)),
        data((uint64_t *)alloc64B(nbytes)) {
    memset(data, 0, nbytes);
  }
  ~Bitset() { free(data); }
  void set(int i) {
    data[i / block_size] |= (Block(1) << (i & (block_size - 1)));
  }
  bool get(int i) {
    return (data[i / block_size] >> (i & (block_size - 1))) & 1;
  }

  void *block_address(int i) { return data + i / block_size; }
};

template <typename dist_t = float> struct Neighbor {
  int id;
  dist_t distance;

  Neighbor() = default;
  Neighbor(int id, dist_t distance) : id(id), distance(distance) {}

  inline friend bool operator<(const Neighbor &lhs, const Neighbor &rhs) {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.id < rhs.id);
  }
  inline friend bool operator>(const Neighbor &lhs, const Neighbor &rhs) {
    return !(lhs < rhs);
  }
};

template <typename dist_t> struct MaxHeap {
  explicit MaxHeap(int capacity) : capacity(capacity), pool(capacity) {}
  void push(int u, dist_t dist) {
    if (size < capacity) {
      pool[size] = {u, dist};
      std::push_heap(pool.begin(), pool.begin() + ++size);
    } else if (dist < pool[0].distance) {
      sift_down(0, u, dist);
    }
  }
  int pop() {
    std::pop_heap(pool.begin(), pool.begin() + size--);
    return pool[size].id;
  }
  void sift_down(int i, int u, dist_t dist) {
    pool[0] = {u, dist};
    for (; 2 * i + 1 < size;) {
      int j = i;
      int l = 2 * i + 1, r = 2 * i + 2;
      if (pool[l].distance > dist) {
        j = l;
      }
      if (r < size && pool[r].distance > std::max(pool[l].distance, dist)) {
        j = r;
      }
      if (i == j) {
        break;
      }
      pool[i] = pool[j];
      i = j;
    }
    pool[i] = {u, dist};
  }
  int size = 0, capacity;
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> pool;
};

template <typename dist_t> struct MinMaxHeap {
  explicit MinMaxHeap(int capacity) : capacity(capacity), pool(capacity) {}
  bool push(int u, dist_t dist) {
    if (cur == capacity) {
      if (dist >= pool[0].distance) {
        return false;
      }
      if (pool[0].id >= 0) {
        size--;
      }
      std::pop_heap(pool.begin(), pool.begin() + cur--);
    }
    pool[cur] = {u, dist};
    std::push_heap(pool.begin(), pool.begin() + ++cur);
    size++;
    return true;
  }
  dist_t max() { return pool[0].distance; }
  void clear() { size = cur = 0; }

  int pop_min() {
    int i = cur - 1;
    for (; i >= 0 && pool[i].id == -1; --i)
      ;
    if (i == -1) {
      return -1;
    }
    int imin = i;
    dist_t vmin = pool[i].distance;
    for (; --i >= 0;) {
      if (pool[i].id != -1 && pool[i].distance < vmin) {
        vmin = pool[i].distance;
        imin = i;
      }
    }
    int ret = pool[imin].id;
    pool[imin].id = -1;
    --size;
    return ret;
  }

  int size = 0, cur = 0, capacity;
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> pool;
};

// 以升序存储距离元素的线性池
template <typename dist_t> struct LinearPool {
  LinearPool(int n, int capacity, int = 0)
      : nb(n), capacity_(capacity), data_(capacity_ + 1), vis(n) {}

  // 二分法查找距离 dist 在池子中的索引
  int find_bsearch(dist_t dist) {
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (data_[mid].distance > dist) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  // 将编号为 u 的数据插入到该池子中，u 到查询点 q 的距离为 dist
  // 插入后池子中的元素保持按照距离的升序排序
  // 插入后位置指针指向第一个未被访问过的元素
  bool insert(int u, dist_t dist) {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo],
                 (size_ - lo) * sizeof(Neighbor<dist_t>));
    data_[lo] = {u, dist};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
    return true;
  }

  int pop() {
    set_checked(data_[cur_].id);
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id)) {
      cur_++;
    }
    return get_id(data_[pre].id);
  }

  bool has_next() const { return cur_ < size_; }
  int id(int i) const { return get_id(data_[i].id); }
  int size() const { return size_; }
  int capacity() const { return capacity_; }

  constexpr static int kMask = 2147483647;
  int get_id(int id) const { return id & kMask; }
  void set_checked(int &id) { id |= 1 << 31; }
  bool is_checked(int id) { return id >> 31 & 1; }

  int nb;                   // 数据集大小
  int size_ = 0;            // 当前的数据量
  int cur_ = 0;             // 当前位置指针
  int capacity_;            // 最大数据量
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> data_;
  Bitset<uint64_t> vis;     // 数据是否被访问过
};

template <typename dist_t> struct HeapPool {
  HeapPool(int n, int capacity, int topk)
      : nb(n), capacity_(capacity), candidates(capacity), retset(topk), vis(n) {
  }
  bool insert(int u, dist_t dist) {
    retset.push(u, dist);
    return candidates.push(u, dist);
  }
  int pop() { return candidates.pop_min(); }
  bool has_next() const { return candidates.size > 0; }
  int id(int i) const { return retset.pool[i].id; }
  int capacity() const { return capacity_; }
  int nb, size_ = 0, capacity_;
  MinMaxHeap<dist_t> candidates;
  MaxHeap<dist_t> retset;
  Bitset<uint64_t> vis;
};

} // namespace searcher

struct Neighbor {
  int id;
  float distance;
  bool flag;

  Neighbor() = default;
  Neighbor(int id, float distance, bool f)
      : id(id), distance(distance), flag(f) {}

  inline bool operator<(const Neighbor &other) const {
    return distance < other.distance;
  }
};

struct Node {
  int id;
  float distance;

  Node() = default;
  Node(int id, float distance) : id(id), distance(distance) {}

  inline bool operator<(const Node &other) const {
    return distance < other.distance;
  }
};

inline int insert_into_pool(Neighbor *addr, int K, Neighbor nn) {
  // find the location to insert
  int left = 0, right = K - 1;
  if (addr[left].distance > nn.distance) {
    memmove(&addr[left + 1], &addr[left], K * sizeof(Neighbor));
    addr[left] = nn;
    return left;
  }
  if (addr[right].distance < nn.distance) {
    addr[K] = nn;
    return K;
  }
  while (left < right - 1) {
    int mid = (left + right) / 2;
    if (addr[mid].distance > nn.distance) {
      right = mid;
    } else {
      left = mid;
    }
  }
  // check equal ID

  while (left > 0) {
    if (addr[left].distance < nn.distance) {
      break;
    }
    if (addr[left].id == nn.id) {
      return K + 1;
    }
    left--;
  }
  if (addr[left].id == nn.id || addr[right].id == nn.id) {
    return K + 1;
  }
  memmove(&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
  addr[right] = nn;
  return right;
}

} // namespace hnsw
/* =================================================================================================== */
/*                                             utils.hpp                                               */
#include <algorithm>
#include <random>
#include <unordered_set>

namespace hnsw {

// 随机生成一个长度为size，元素的取值范围是[0, N - 1]的，按照严格升序排序的int类型数组
inline void GenRandom(std::mt19937 &rng, int *addr, const int size,
                      const int N) {
  for (int i = 0; i < size; ++i) {
    addr[i] = rng() % (N - size);
  }
  std::sort(addr, addr + size);
  for (int i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  int off = rng() % N;
  for (int i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % N;
  }
}

struct RandomGenerator {
  std::mt19937 mt;

  explicit RandomGenerator(int64_t seed = 1234) : mt((unsigned int)seed) {}

  /// random positive integer
  int rand_int() { return mt() & 0x7fffffff; }

  /// random int64_t
  int64_t rand_int64() {
    return int64_t(rand_int()) | int64_t(rand_int()) << 31;
  }

  /// generate random integer between 0 and max-1
  int rand_int(int max) { return rand_int() % max; }
};

} // namespace hnsw
/* =================================================================================================== */
/*                                          simd/avx2.hpp                                              */
#if defined(__AVX2__)

#include <cstdint>
#include <immintrin.h>

namespace hnsw {

inline float reduce_add_f32x8(__m256 x) {
  auto sumh =
      _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

inline int32_t reduce_add_i32x8(__m256i x) {
  auto sumh =
      _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
  auto tmp2 = _mm_hadd_epi32(sumh, sumh);
  return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

inline int32_t reduce_add_i16x16(__m256i x) {
  auto sumh = _mm_add_epi16(_mm256_extracti128_si256(x, 0),
                            _mm256_extracti128_si256(x, 1));
  auto tmp = _mm256_cvtepi16_epi32(sumh);
  auto sumhh = _mm_add_epi32(_mm256_extracti128_si256(tmp, 0),
                             _mm256_extracti128_si256(tmp, 1));
  auto tmp2 = _mm_hadd_epi32(sumhh, sumhh);
  return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

} // namespace hnsw

#endif
/* =================================================================================================== */
/*                                           simd/avx512.hpp                                           */
#if defined(__AVX512F__)

#include <cstdint>
#include <immintrin.h>

namespace hnsw {

inline float reduce_add_f32x16(__m512 x) {
  auto sumh =
      _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1));
  auto sumhh =
      _mm_add_ps(_mm256_castps256_ps128(sumh), _mm256_extractf128_ps(sumh, 1));
  auto tmp1 = _mm_add_ps(sumhh, _mm_movehl_ps(sumhh, sumhh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

inline int32_t reduce_add_i32x16(__m512i x) {
  auto sumh = _mm256_add_epi32(_mm512_extracti32x8_epi32(x, 0),
                               _mm512_extracti32x8_epi32(x, 1));
  auto sumhh = _mm_add_epi32(_mm256_castsi256_si128(sumh),
                             _mm256_extracti128_si256(sumh, 1));
  auto tmp2 = _mm_hadd_epi32(sumhh, sumhh);
  return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

} // namespace hnsw

#endif
/* =================================================================================================== */
/*                                        simd/distance.hpp                                            */
#include <cstdint>
#include <cstdio>
#if defined(__SSE2__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace hnsw {

template <typename T1, typename T2, typename U, typename... Params>
using Dist = U (*)(const T1 *, const T2 *, int, Params...);

GLASS_INLINE inline void prefetch_L1(const void *address) {
#if defined(__SSE2__)
  _mm_prefetch((const char *)address, _MM_HINT_T0);
#else
  __builtin_prefetch(address, 0, 3);
#endif
}

GLASS_INLINE inline void prefetch_L2(const void *address) {
#if defined(__SSE2__)
  _mm_prefetch((const char *)address, _MM_HINT_T1);
#else
  __builtin_prefetch(address, 0, 2);
#endif
}

GLASS_INLINE inline void prefetch_L3(const void *address) {
#if defined(__SSE2__)
  _mm_prefetch((const char *)address, _MM_HINT_T2);
#else
  __builtin_prefetch(address, 0, 1);
#endif
}

inline void mem_prefetch(char *ptr, const int num_lines) {
  switch (num_lines) {
  default:
    [[fallthrough]];
  case 28:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 27:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 26:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 25:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 24:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 23:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 22:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 21:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 20:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 19:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 18:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 17:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 16:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 15:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 14:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 13:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 12:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 11:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 10:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 9:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 8:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 7:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 6:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 5:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 4:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 3:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 2:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 1:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 0:
    break;
  }
}

FAST_BEGIN
inline float L2SqrRef(const float *x, const float *y, int d) {
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return sum;
}
FAST_END

FAST_BEGIN
inline float IPRef(const float *x, const float *y, int d) {
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += x[i] * y[i];
  }
  return sum;
}
FAST_END

inline float L2Sqr(const float *x, const float *y, int d) {
#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    auto xx = _mm512_loadu_ps(x);
    x += 16;
    auto yy = _mm512_loadu_ps(y);
    y += 16;
    auto t = _mm512_sub_ps(xx, yy);
    sum = _mm512_add_ps(sum, _mm512_mul_ps(t, t));
  }
  return reduce_add_f32x16(sum);
#elif defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    auto xx = _mm256_loadu_ps(x);
    x += 8;
    auto yy = _mm256_loadu_ps(y);
    y += 8;
    auto t = _mm256_sub_ps(xx, yy);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(t, t));
  }
  return reduce_add_f32x8(sum);
#elif defined(__aarch64__)
  float32x4_t sum = vdupq_n_f32(0);
  for (int32_t i = 0; i < d; i += 4) {
    auto xx = vld1q_f32(x + i);
    auto yy = vld1q_f32(y + i);
    auto t = vsubq_f32(xx, yy);
    sum = vmlaq_f32(sum, t, t);
  }
  return vaddvq_f32(sum);
#else
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return sum;
#endif
}

inline float IP(const float *x, const float *y, int d) {
#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    auto xx = _mm512_loadu_ps(x);
    x += 16;
    auto yy = _mm512_loadu_ps(y);
    y += 16;
    sum = _mm512_add_ps(sum, _mm512_mul_ps(xx, yy));
  }
  return -reduce_add_f32x16(sum);
#elif defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    auto xx = _mm256_loadu_ps(x);
    x += 8;
    auto yy = _mm256_loadu_ps(y);
    y += 8;
    sum = _mm256_add_ps(sum, _mm256_mul_ps(xx, yy));
  }
  return -reduce_add_f32x8(sum);
#elif defined(__aarch64__)
  float32x4_t sum = vdupq_n_f32(0);
  for (int32_t i = 0; i < d; i += 4) {
    auto xx = vld1q_f32(x + i);
    auto yy = vld1q_f32(y + i);
    sum = vmlaq_f32(sum, xx, yy);
  }
  return vaddvq_f32(sum);
#else
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    sum += x[i] * y[i];
  }
  return -sum;
#endif
}

inline float L2SqrSQ8_ext(const float *x, const uint8_t *y, int d,
                          const float *mi, const float *dif) {
#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  __m512 dot5 = _mm512_set1_ps(0.5f);
  __m512 const_255 = _mm512_set1_ps(255.0f);
  for (int i = 0; i < d; i += 16) {
    auto zz = _mm_loadu_epi8(y + i);
    auto zzz = _mm512_cvtepu8_epi32(zz);
    auto yy = _mm512_cvtepi32_ps(zzz);
    yy = _mm512_add_ps(yy, dot5);
    auto mi512 = _mm512_loadu_ps(mi + i);
    auto dif512 = _mm512_loadu_ps(dif + i);
    yy = _mm512_mul_ps(yy, dif512);
    yy = _mm512_add_ps(yy, _mm512_mul_ps(mi512, const_255));
    auto xx = _mm512_loadu_ps(x + i);
    auto d = _mm512_sub_ps(_mm512_mul_ps(xx, const_255), yy);
    sum = _mm512_fmadd_ps(d, d, sum);
  }
  return reduce_add_f32x16(sum);
#else
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = (y[i] + 0.5f);
    yy = yy * dif[i] + mi[i] * 255.0f;
    auto dif = x[i] * 255.0f - yy;
    sum += dif * dif;
  }
  return sum;
#endif
}

inline float IPSQ8_ext(const float *x, const uint8_t *y, int d, const float *mi,
                       const float *dif) {

#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  __m512 dot5 = _mm512_set1_ps(0.5f);
  __m512 const_255 = _mm512_set1_ps(255.0f);
  for (int i = 0; i < d; i += 16) {
    auto zz = _mm_loadu_epi8(y + i);
    auto zzz = _mm512_cvtepu8_epi32(zz);
    auto yy = _mm512_cvtepi32_ps(zzz);
    yy = _mm512_add_ps(yy, dot5);
    auto mi512 = _mm512_loadu_ps(mi + i);
    auto dif512 = _mm512_loadu_ps(dif + i);
    yy = _mm512_mul_ps(yy, dif512);
    yy = _mm512_add_ps(yy, _mm512_mul_ps(mi512, const_255));
    auto xx = _mm512_loadu_ps(x + i);
    sum = _mm512_fmadd_ps(xx, yy, sum);
  }
  return -reduce_add_f32x16(sum);
#else
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = y[i] + 0.5f;
    yy = yy * dif[i] + mi[i] * 255.0f;
    sum += x[i] * yy;
  }
  return -sum;
#endif
}

inline int32_t L2SqrSQ4(const uint8_t *x, const uint8_t *y, int d) {
#if defined(__AVX2__)
  __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
  __m256i mask = _mm256_set1_epi8(0xf);
  for (int i = 0; i < d; i += 64) {
    auto xx = _mm256_loadu_si256((__m256i *)(x + i / 2));
    auto yy = _mm256_loadu_si256((__m256i *)(y + i / 2));
    auto xx1 = _mm256_and_si256(xx, mask);
    auto xx2 = _mm256_and_si256(_mm256_srli_epi16(xx, 4), mask);
    auto yy1 = _mm256_and_si256(yy, mask);
    auto yy2 = _mm256_and_si256(_mm256_srli_epi16(yy, 4), mask);
    auto d1 = _mm256_sub_epi8(xx1, yy1);
    auto d2 = _mm256_sub_epi8(xx2, yy2);
    d1 = _mm256_abs_epi8(d1);
    d2 = _mm256_abs_epi8(d2);
    sum1 = _mm256_add_epi16(sum1, _mm256_maddubs_epi16(d1, d1));
    sum2 = _mm256_add_epi16(sum2, _mm256_maddubs_epi16(d2, d2));
  }
  sum1 = _mm256_add_epi32(sum1, sum2);
  return reduce_add_i16x16(sum1);
#else
  int32_t sum = 0;
  for (int i = 0; i < d; ++i) {
    {
      int32_t xx = x[i / 2] & 15;
      int32_t yy = y[i / 2] & 15;
      sum += (xx - yy) * (xx - yy);
    }
    {
      int32_t xx = x[i / 2] >> 4 & 15;
      int32_t yy = y[i / 2] >> 4 & 15;
      sum += (xx - yy) * (xx - yy);
    }
  }
  return sum;
#endif
}

} // namespace hnsw
/* =================================================================================================== */
/*                                        quant/fp32_quant.hpp                                         */
namespace hnsw {

template <Metric metric, int DIM = 0> struct FP32Quantizer {
  using data_type = float;
  constexpr static int kAlign = 16;
  int d, d_align;
  int64_t code_size;
  char *codes = nullptr;

  FP32Quantizer() = default;

  explicit FP32Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align * 4) {}

  ~FP32Quantizer() { free(codes); }

  void train(const float *data, int64_t n) {
    codes = (char *)alloc2M(n * code_size);
    for (int64_t i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
  }

  void encode(const float *from, char *to) { std::memcpy(to, from, d * 4); }

  char *get_data(int u) const { return codes + u * code_size; }

  template <typename Pool>
  void reorder(const Pool &pool, const float *, int *dst, int k) const {
    for (int i = 0; i < k; ++i) {
      dst[i] = pool.id(i);
    }
  }

  template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
    using dist_type = float;
    constexpr static auto dist_func = metric == Metric::L2 ? L2Sqr : IP;
    const FP32Quantizer &quant;
    float *q = nullptr;
    Computer(const FP32Quantizer &quant, const float *query)
        : quant(quant), q((float *)alloc64B(quant.d_align * 4)) {
      std::memcpy(q, query, quant.d * 4);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), quant.d);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query) const {
    return Computer<0>(*this, query);
  }
};

} // namespace hnsw
/* =================================================================================================== */
/*                                        quant/sq4_quant.hpp                                          */
#include <cmath>

namespace hnsw {

template <Metric metric, typename Reorderer = FP32Quantizer<metric>,
          int DIM = 0>
struct SQ4Quantizer {
  using data_type = uint8_t;
  constexpr static int kAlign = 128;
  float mx = -HUGE_VALF, mi = HUGE_VALF, dif;
  int d, d_align;
  int64_t code_size;
  data_type *codes = nullptr;

  Reorderer reorderer;

  SQ4Quantizer() = default;

  explicit SQ4Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align / 2),
        reorderer(dim) {}

  ~SQ4Quantizer() { free(codes); }

  void train(const float *data, int n) {
    for (int64_t i = 0; i < n * d; ++i) {
      mx = std::max(mx, data[i]);
      mi = std::min(mi, data[i]);
    }
    dif = mx - mi;
    codes = (data_type *)alloc2M(n * code_size);
    for (int i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
    reorderer.train(data, n);
  }

  char *get_data(int u) const { return (char *)codes + u * code_size; }

  void encode(const float *from, char *to) const {
    for (int j = 0; j < d; ++j) {
      float x = (from[j] - mi) / dif;
      if (x < 0.0) {
        x = 0.0;
      }
      if (x > 0.999) {
        x = 0.999;
      }
      uint8_t y = 16 * x;
      if (j & 1) {
        to[j / 2] |= y << 4;
      } else {
        to[j / 2] |= y;
      }
    }
  }

  template <typename Pool>
  void reorder(const Pool &pool, const float *q, int *dst, int k) const {
    int cap = pool.capacity();
    auto computer = reorderer.get_computer(q);
    searcher::MaxHeap<typename Reorderer::template Computer<0>::dist_type> heap(
        k);
    for (int i = 0; i < cap; ++i) {
      if (i + 1 < cap) {
        computer.prefetch(pool.id(i + 1), 1);
      }
      int id = pool.id(i);
      float dist = computer(id);
      heap.push(id, dist);
    }
    for (int i = 0; i < k; ++i) {
      dst[i] = heap.pop();
    }
  }

  template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
    using dist_type = int32_t;
    constexpr static auto dist_func = L2SqrSQ4;
    const SQ4Quantizer &quant;
    uint8_t *q;
    Computer(const SQ4Quantizer &quant, const float *query)
        : quant(quant), q((uint8_t *)alloc64B(quant.code_size)) {
      quant.encode(query, (char *)q);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), quant.d_align);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query) const {
    return Computer<0>(*this, query);
  }
};

} // namespace hnsw
/* =================================================================================================== */
/*                                         quant/sq8_quant.hpp                                         */
#include <cmath>
#include <vector>

namespace hnsw {

template <Metric metric, int DIM = 0> struct SQ8Quantizer {
  using data_type = uint8_t;
  constexpr static int kAlign = 16;
  int d, d_align;
  int64_t code_size;
  char *codes = nullptr;
  std::vector<float> mx, mi, dif;

  SQ8Quantizer() = default;

  explicit SQ8Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align),
        mx(d_align, -HUGE_VALF), mi(d_align, HUGE_VALF), dif(d_align) {}

  ~SQ8Quantizer() { free(codes); }

  void train(const float *data, int n) {
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < d; ++j) {
        mx[j] = std::max(mx[j], data[i * d + j]);
        mi[j] = std::min(mi[j], data[i * d + j]);
      }
    }
    for (int64_t j = 0; j < d; ++j) {
      dif[j] = mx[j] - mi[j];
    }
    for (int64_t j = d; j < d_align; ++j) {
      dif[j] = mx[j] = mi[j] = 0;
    }
    codes = (char *)alloc2M((size_t)n * code_size);
    for (int i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
  }

  char *get_data(int u) const { return codes + u * code_size; }

  void encode(const float *from, char *to) const {
    for (int j = 0; j < d; ++j) {
      float x = (from[j] - mi[j]) / dif[j];
      if (x < 0) {
        x = 0.0;
      }
      if (x > 1.0) {
        x = 1.0;
      }
      uint8_t y = x * 255;
      to[j] = y;
    }
  }

  template <typename Pool>
  void reorder(const Pool &pool, const float * /**q*/, int *dst, int k) const {
    for (int i = 0; i < k; ++i) {
      dst[i] = pool.id(i);
    }
  }

  template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
    using dist_type = float;
    constexpr static auto dist_func =
        metric == Metric::L2 ? L2SqrSQ8_ext : IPSQ8_ext;
    const SQ8Quantizer &quant;
    float *q;
    const float *mi, *dif;
    Computer(const SQ8Quantizer &quant, const float *query)
        : quant(quant), q((float *)alloc64B(quant.d_align * 4)),
          mi(quant.mi.data()), dif(quant.dif.data()) {
      std::memcpy(q, query, quant.d * 4);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), quant.d_align, mi,
                       dif);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query) const {
    return Computer<0>(*this, query);
  }
};

} // namespace hnsw
/* =================================================================================================== */
/*                                          quant/quant.hpp                                            */
#include <string>
#include <unordered_map>

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
/*=====================================================================================================*/
/*                                     hnsw/HNSWInitializer.hpp                                        */
#include <cstdlib>
#include <fstream>
#include <vector>

namespace hnsw {

struct HNSWInitializer {
  int N, K;
  int ep;
  std::vector<int> levels;
  std::vector<std::vector<int, align_alloc<int>>> lists;
  HNSWInitializer() = default;

  explicit HNSWInitializer(int n, int K = 0)
      : N(n), K(K), levels(n), lists(n) {}

  HNSWInitializer(const HNSWInitializer &rhs) = default;

  int at(int level, int u, int i) const {
    return lists[u][(level - 1) * K + i];
  }

  int &at(int level, int u, int i) { return lists[u][(level - 1) * K + i]; }

  const int *edges(int level, int u) const {
    return lists[u].data() + (level - 1) * K;
  }

  int *edges(int level, int u) { return lists[u].data() + (level - 1) * K; }

  template <typename Pool, typename Computer>
  void initialize(Pool &pool, const Computer &computer) const {
    int u = ep;
    auto cur_dist = computer(u);
    for (int level = levels[u]; level > 0; --level) {
      bool changed = true;
      while (changed) {
        changed = false;
        const int *list = edges(level, u);
        for (int i = 0; i < K && list[i] != -1; ++i) {
          int v = list[i];
          auto dist = computer(v);
          if (dist < cur_dist) {
            cur_dist = dist;
            u = v;
            changed = true;
          }
        }
      }
    }
    pool.insert(u, cur_dist);
    pool.vis.set(u);
  }

  void load(std::ifstream &reader) {
    reader.read((char *)&N, 4);
    reader.read((char *)&K, 4);
    reader.read((char *)&ep, 4);
    for (int i = 0; i < N; ++i) {
      int cur;
      reader.read((char *)&cur, 4);
      levels[i] = cur / K;
      lists[i].assign(cur, -1);
      reader.read((char *)lists[i].data(), cur * 4);
    }
  }

  void save(std::ofstream &writer) const {
    writer.write((char *)&N, 4);
    writer.write((char *)&K, 4);
    writer.write((char *)&ep, 4);
    for (int i = 0; i < N; ++i) {
      int cur = levels[i] * K;
      writer.write((char *)&cur, 4);
      writer.write((char *)lists[i].data(), cur * 4);
    }
  }
};

} // namespace hnsw
/* =================================================================================================== */
/*                                             graph.hpp                                               */
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

namespace hnsw {

constexpr int EMPTY_ID = -1;

template <typename node_t> struct Graph {
  int N, K;

  node_t *data = nullptr;

  std::unique_ptr<HNSWInitializer> initializer = nullptr;

  std::vector<int> eps;

  Graph() = default;

  Graph(node_t *edges, int N, int K) : N(N), K(K), data(edges) {}

  Graph(int N, int K)
      : N(N), K(K), data((node_t *)alloc2M((size_t)N * K * sizeof(node_t))) {}

  Graph(const Graph &g) : Graph(g.N, g.K) {
    this->eps = g.eps;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < K; ++j) {
        at(i, j) = g.at(i, j);
      }
    }
    if (g.initializer) {
      initializer = std::make_unique<HNSWInitializer>(*g.initializer);
    }
  }

  void init(int N, int K) {
    data = (node_t *)alloc2M((size_t)N * K * sizeof(node_t));
    std::memset(data, -1, N * K * sizeof(node_t));
    this->K = K;
    this->N = N;
  }

  ~Graph() { free(data); }

  const int *edges(int u) const { return data + K * u; }

  int *edges(int u) { return data + K * u; }

  node_t at(int i, int j) const { return data[i * K + j]; }

  node_t &at(int i, int j) { return data[i * K + j]; }

  void prefetch(int u, int lines) const {
    mem_prefetch((char *)edges(u), lines);
  }

  template <typename Pool, typename Computer>
  void initialize_search(Pool &pool, const Computer &computer) const {
    if (initializer) {
      initializer->initialize(pool, computer);
    } else {
      for (auto ep : eps) {
        pool.insert(ep, computer(ep));
      }
    }
  }

  void save(const std::string &filename) const {
    static_assert(std::is_same<node_t, int32_t>::value);
    std::ofstream writer(filename.c_str(), std::ios::binary);
    int nep = eps.size();
    writer.write((char *)&nep, 4);
    writer.write((char *)eps.data(), nep * 4);
    writer.write((char *)&N, 4);
    writer.write((char *)&K, 4);
    writer.write((char *)data, N * K * 4);
    if (initializer) {
      initializer->save(writer);
    }
#ifdef DEBUG_ENABLED
    fprintf(stderr, "Graph Saving done\n");
#endif
  }

  void load(const std::string &filename) {
    static_assert(std::is_same<node_t, int32_t>::value);
    free(data);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    int nep;
    reader.read((char *)&nep, 4);
    eps.resize(nep);
    reader.read((char *)eps.data(), nep * 4);
    reader.read((char *)&N, 4);
    reader.read((char *)&K, 4);
    data = (node_t *)alloc2M((size_t)N * K * 4);
    reader.read((char *)data, N * K * 4);
    if (reader.peek() != EOF) {
      initializer = std::make_unique<HNSWInitializer>(N);
      initializer->load(reader);
    }
#ifdef DEBUG_ENABLED
    fprintf(stderr, "Graph Loding done\n");
#endif
  }
};

} // namespace hnsw
/* =================================================================================================== */
/*                                             build.hpp                                               */
namespace hnsw {

struct Builder {
  virtual void Build(float *data, int nb) = 0;
  virtual hnsw::Graph<int> GetGraph() = 0;
  virtual ~Builder() = default;
};

} // namespace hnsw
/* =================================================================================================== */
/*                                           searcher.hpp                                              */
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace hnsw {

struct SearcherBase {
  virtual void SetData(const float *data, int n, int dim) = 0;
  virtual void Optimize() = 0;
  virtual void Search(const float *q, int k, int *dst) const = 0;
  virtual void SetEf(int ef) = 0;
  virtual ~SearcherBase() = default;
};

template <typename Quantizer> struct Searcher : public SearcherBase {

  int d;
  int nb;
  Graph<int> graph;
  Quantizer quant;

  // Search parameters
  int ef = 32;

  // Memory prefetch parameters
  int po = 1;
  int pl = 1;

  // Optimization parameters
  constexpr static int kOptimizePoints = 1000;
  constexpr static int kTryPos = 10;
  constexpr static int kTryPls = 5;
  constexpr static int kTryK = 10;
  int sample_points_num;
  std::vector<float> optimize_queries;
  const int graph_po;

  Searcher(const Graph<int> &graph) : graph(graph), graph_po(graph.K / 16) {}

  void SetData(const float *data, int n, int dim) override {
    this->nb = n;
    this->d = dim;
    quant = Quantizer(d);
    quant.train(data, n);

    sample_points_num = std::min(kOptimizePoints, nb - 1);
    std::vector<int> sample_points(sample_points_num);
    std::mt19937 rng;
    GenRandom(rng, sample_points.data(), sample_points_num, nb);
    optimize_queries.resize(sample_points_num * d);
    for (int i = 0; i < sample_points_num; ++i) {
      memcpy(optimize_queries.data() + i * d, data + sample_points[i] * d,
             d * sizeof(float));
    }
  }

  void SetEf(int ef) override { this->ef = ef; }

  void Optimize() override {
    std::vector<int> try_pos(std::min(kTryPos, graph.K));
    std::vector<int> try_pls(
        std::min(kTryPls, (int)upper_div(quant.code_size, 64)));
    std::iota(try_pos.begin(), try_pos.end(), 1);
    std::iota(try_pls.begin(), try_pls.end(), 1);
    std::vector<int> dummy_dst(kTryK);
#ifdef DEBUG_ENABLED
    fprintf(stderr, "=============Start optimization=============\n");
#endif
    { // warmup
      for (int i = 0; i < sample_points_num; ++i) {
        Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
      }
    }

    float min_ela = std::numeric_limits<float>::max();
    int best_po = 0, best_pl = 0;
    for (auto try_po : try_pos) {
      for (auto try_pl : try_pls) {
        this->po = try_po;
        this->pl = try_pl;
        auto st = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < sample_points_num; ++i) {
          Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
        }

        auto ed = std::chrono::high_resolution_clock::now();
        auto ela = std::chrono::duration<double>(ed - st).count();
        if (ela < min_ela) {
          min_ela = ela;
          best_po = try_po;
          best_pl = try_pl;
        }
      }
    }
    this->po = 1;
    this->pl = 1;
#ifdef DEBUG_ENABLED
    auto st = std::chrono::high_resolution_clock::now();
#endif
    for (int i = 0; i < sample_points_num; ++i) {
      Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
    }
#ifdef DEBUG_ENABLED
    auto ed = std::chrono::high_resolution_clock::now();
    float baseline_ela = std::chrono::duration<double>(ed - st).count();
    fprintf(stderr, "settint best po = %d, best pl = %d\n"
           "gaining %.2f%% performance improvement\n============="
           "Done optimization=============\n",
           best_po, best_pl, 100.0 * (baseline_ela / min_ela - 1));
#endif
    this->po = best_po;
    this->pl = best_pl;
  }

  void Search(const float *q, int k, int *dst) const override {
    auto computer = quant.get_computer(q);
    searcher::LinearPool<typename Quantizer::template Computer<0>::dist_type> pool(nb, std::max(k, ef), k);
    // searcher::HeapPool<typename Quantizer::template Computer<0>::dist_type> pool(nb, std::max(k, ef), k);
    graph.initialize_search(pool, computer);
    SearchImpl(pool, computer);
    quant.reorder(pool, q, dst, k);
  }

  // pool 存储 k 个最近邻节点的候选节点，其大小(记为 l )会大于 k
  // 每次都从 pool 中取出第一个还没有被访问过的节点
  // 将该节点标记为已访问，将其所有邻居节点都加入 pool
  // 计算 pool 中所有节点距离要查询的数据 q 的距离， 按照距离的升序对 pool 中的节点进行排序
  // 如果此时 pool 的元素个数超过了 l， 将多余的节点删除
  template <typename Pool, typename Computer>
  void SearchImpl(Pool &pool, const Computer &computer) const {
#ifdef DEBUG_ENABLED
    int compute_num = 0;
#endif
    while (pool.has_next()) {
      auto u = pool.pop();
      graph.prefetch(u, graph_po);
      for (int i = 0; i < po; ++i) {
        int to = graph.at(u, i);
        computer.prefetch(to, pl);
      }
      for (int i = 0; i < graph.K; ++i) {
        int v = graph.at(u, i);
        if (v == EMPTY_ID) {
          break;
        }
        if (pool.vis.get(v)) {
          continue;
        }
        pool.vis.set(v);
        if (i + po < graph.K && graph.at(u, i + po) != -1) {
          int to = graph.at(u, i + po);
          computer.prefetch(to, pl);
        }
        auto cur_dist = computer(v);
        pool.insert(v, cur_dist);
#ifdef DEBUG_ENABLED
        compute_num ++;
#endif
      }
    }
#ifdef DEBUG_ENABLED
    fprintf(stderr, "Compute number: %d\n", compute_num);
#endif
  }
};

inline std::unique_ptr<SearcherBase> create_searcher(const Graph<int> &graph,
                                                     const std::string &metric,
                                                     int level = 1) {
  auto m = metric_map[metric];
  if (level == 0) {
    if (m == Metric::L2) {
      return std::make_unique<Searcher<FP32Quantizer<Metric::L2>>>(graph);
    } else if (m == Metric::IP) {
      return std::make_unique<Searcher<FP32Quantizer<Metric::IP>>>(graph);
    } else {
#ifdef DEBUG_ENABLED
      fprintf(stderr, "Metric not suppported\n");
#endif
      return nullptr;
    }
  } else if (level == 1) {
    if (m == Metric::L2) {
      return std::make_unique<Searcher<SQ8Quantizer<Metric::L2>>>(graph);
    } else if (m == Metric::IP) {
      return std::make_unique<Searcher<SQ8Quantizer<Metric::IP>>>(graph);
    } else {
#ifdef DEBUG_ENABLED
      fprintf(stderr, "Metric not suppported\n");
#endif
      return nullptr;
    }
  } else if (level == 2) {
    if (m == Metric::L2) {
      return std::make_unique<Searcher<SQ4Quantizer<Metric::L2>>>(graph);
    } else if (m == Metric::IP) {
      return std::make_unique<Searcher<SQ4Quantizer<Metric::IP>>>(graph);
    } else {
#ifdef DEBUG_ENABLED
      fprintf(stderr, "Metric not suppported\n");
#endif
      return nullptr;
    }
  } else {
#ifdef DEBUG_ENABLED
    fprintf(stderr, "Quantizer type not supported\n");
#endif
    return nullptr;
  }
}

} // namespace hnsw
/* =================================================================================================== */
/*                                     hnswlib/visited_list_pool.h                                     */
#include <string.h>
#include <deque>

namespace hnswlib {
typedef unsigned short int vl_type;

class VisitedList {
 public:
    vl_type curV;
    vl_type *mass;
    unsigned int numelements;

    VisitedList(int numelements1) {
        curV = -1;
        numelements = numelements1;
        mass = new vl_type[numelements];
    }

    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    ~VisitedList() { delete[] mass; }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    std::deque<VisitedList *> pool;
    int numelements;

 public:
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements));
    }

    VisitedList *getFreeVisitedList() {
        VisitedList *rez;
        {
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements);
            }
        }
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList *vl) {
        pool.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};
}  // namespace hnswlib
/* =================================================================================================== */
/*                                        hnswlib/hnswlib.h                                            */
#ifndef NO_MANUAL_VECTORIZATION
#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
void cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
  __cpuidex(out, eax, ecx);
}
static __int64 xgetbv(unsigned int x) { return _xgetbv(x); }
#else
#include <cpuid.h>
#include <stdint.h>
#include <x86intrin.h>
static void cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx) {
  __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
}
static uint64_t xgetbv(unsigned int index) {
  uint32_t eax, edx;
  __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
  return ((uint64_t)edx << 32) | eax;
}
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

// Adapted from https://github.com/Mysticial/FeatureDetector
#define _XCR_XFEATURE_ENABLED_MASK 0

[[maybe_unused]] static bool AVXCapable() {
  int cpuInfo[4];

  // CPU support
  cpuid(cpuInfo, 0, 0);
  int nIds = cpuInfo[0];

  bool HW_AVX = false;
  if (nIds >= 0x00000001) {
    cpuid(cpuInfo, 0x00000001, 0);
    HW_AVX = (cpuInfo[2] & ((int)1 << 28)) != 0;
  }

  // OS support
  cpuid(cpuInfo, 1, 0);

  bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
  bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

  bool avxSupported = false;
  if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
    uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    avxSupported = (xcrFeatureMask & 0x6) == 0x6;
  }
  return HW_AVX && avxSupported;
}

[[maybe_unused]] static bool AVX512Capable() {
  if (!AVXCapable())
    return false;

  int cpuInfo[4];

  // CPU support
  cpuid(cpuInfo, 0, 0);
  int nIds = cpuInfo[0];

  bool HW_AVX512F = false;
  if (nIds >= 0x00000007) { //  AVX512 Foundation
    cpuid(cpuInfo, 0x00000007, 0);
    HW_AVX512F = (cpuInfo[1] & ((int)1 << 16)) != 0;
  }

  // OS support
  cpuid(cpuInfo, 1, 0);

  bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
  bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

  bool avx512Supported = false;
  if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
    uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;
  }
  return HW_AVX512F && avx512Supported;
}
#endif

#include <iostream>
#include <queue>
#include <string.h>
#include <vector>

namespace hnswlib {
typedef size_t labeltype;

// This can be extended to store state for filtering (e.g. from a std::set)
class BaseFilterFunctor {
public:
  virtual bool operator()(hnswlib::labeltype) { return true; }
};

template <typename T> class pairGreater {
public:
  bool operator()(const T &p1, const T &p2) { return p1.first > p2.first; }
};

template <typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
  out.write((char *)&podRef, sizeof(T));
}

template <typename T> static void readBinaryPOD(std::istream &in, T &podRef) {
  in.read((char *)&podRef, sizeof(T));
}

template <typename MTYPE>
using DISTFUNC = MTYPE (*)(const void *, const void *, const void *);

template <typename MTYPE> class SpaceInterface {
public:
  // virtual void search(void *);
  virtual size_t get_data_size() = 0;

  virtual DISTFUNC<MTYPE> get_dist_func() = 0;

  virtual void *get_dist_func_param() = 0;

  virtual ~SpaceInterface() {}
};

template <typename dist_t> class AlgorithmInterface {
public:
  virtual void addPoint(const void *datapoint, labeltype label,
                        bool replace_deleted = false) = 0;

  virtual std::priority_queue<std::pair<dist_t, labeltype>>
  searchKnn(const void *, size_t,
            BaseFilterFunctor *isIdAllowed = nullptr) const = 0;

  // Return k nearest neighbor in the order of closer fist
  virtual std::vector<std::pair<dist_t, labeltype>>
  searchKnnCloserFirst(const void *query_data, size_t k,
                       BaseFilterFunctor *isIdAllowed = nullptr) const;

  virtual void saveIndex(const std::string &location) = 0;
  virtual ~AlgorithmInterface() {}
};

template <typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
AlgorithmInterface<dist_t>::searchKnnCloserFirst(
    const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed) const {
  std::vector<std::pair<dist_t, labeltype>> result;

  // here searchKnn returns the result in the order of further first
  auto ret = searchKnn(query_data, k, isIdAllowed);
  {
    size_t sz = ret.size();
    result.resize(sz);
    while (!ret.empty()) {
      result[--sz] = ret.top();
      ret.pop();
    }
  }

  return result;
}
} // namespace hnswlib
/* =================================================================================================== */
/*                                        hnswlib/hnswalg.h                                            */
#include <assert.h>
#include <list>
#include <random>
#include <stdlib.h>
#include <unordered_map>
#include <unordered_set>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
public:
  static const unsigned char DELETE_MARK = 0x01;

  size_t max_elements_{0};                      // 数据量
  mutable size_t cur_element_count = 0;   // current number of elements
  size_t size_data_per_element_{0};             // 每个元素的byte大小
  size_t size_links_per_element_{0};
  mutable size_t num_deleted_ = 0;  // number of deleted elements
  size_t M_{0};
  size_t maxM_{0};
  size_t maxM0_{0};
  size_t ef_construction_{0};
  size_t ef_{0};

  double mult_{0.0}, revSize_{0.0};
  int maxlevel_{0};

  VisitedListPool *visited_list_pool_{nullptr};

  tableint enterpoint_node_{0};

  size_t size_links_level0_{0};
  size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

  char *data_level0_memory_{nullptr};
  char **linkLists_{nullptr};
  std::vector<int> element_levels_; // keeps level of each element

  size_t data_size_{0};

  DISTFUNC<dist_t> fstdistfunc_;          // 距离度量函数
  void *dist_func_param_{nullptr};

  std::unordered_map<labeltype, tableint> label_lookup_;

  std::default_random_engine level_generator_;
  std::default_random_engine update_probability_generator_;

  bool allow_replace_deleted_ = false; // flag to replace deleted elements
                                       // (marked as deleted) during insertions

  std::unordered_set<tableint> deleted_elements; // contains internal ids of deleted elements

  HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location,
                  bool /**nmslib*/ = false, size_t max_elements = 0,
                  bool allow_replace_deleted = false)
      : allow_replace_deleted_(allow_replace_deleted) {
    loadIndex(location, s, max_elements);
  }

  HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16,
                  size_t ef_construction = 200, size_t random_seed = 100,
                  bool allow_replace_deleted = false)
      : element_levels_(max_elements),
        allow_replace_deleted_(allow_replace_deleted) {
    max_elements_ = max_elements;
    num_deleted_ = 0;
    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    M_ = M;
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    size_data_per_element_ =
        size_links_level0_ + data_size_ + sizeof(labeltype);
    offsetData_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
    offsetLevel0_ = 0;

    data_level0_memory_ =
        (char *)malloc(max_elements_ * size_data_per_element_);

    cur_element_count = 0;

    visited_list_pool_ = new VisitedListPool(1, max_elements);

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
    size_links_per_element_ =
        maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
    mult_ = 1 / log(1.0 * M_);
    revSize_ = 1.0 / mult_;
  }

  ~HierarchicalNSW() {
    free(data_level0_memory_);
    for (tableint i = 0; i < cur_element_count; i++) {
      if (element_levels_[i] > 0)
        free(linkLists_[i]);
    }
    free(linkLists_);
    delete visited_list_pool_;
  }

  struct CompareByFirst {
    constexpr bool
    operator()(std::pair<dist_t, tableint> const &a,
               std::pair<dist_t, tableint> const &b) const noexcept {
      return a.first < b.first;
    }
  };

  void setEf(size_t ef) { ef_ = ef; }

  inline labeltype getExternalLabel(tableint internal_id) const {
    labeltype return_label;
    memcpy(&return_label,
           (data_level0_memory_ + internal_id * size_data_per_element_ +
            label_offset_),
           sizeof(labeltype));
    return return_label;
  }

  inline void setExternalLabel(tableint internal_id, labeltype label) const {
    memcpy((data_level0_memory_ + internal_id * size_data_per_element_ +
            label_offset_),
           &label, sizeof(labeltype));
  }

  inline labeltype *getExternalLabeLp(tableint internal_id) const {
    return (labeltype *)(data_level0_memory_ +
                         internal_id * size_data_per_element_ + label_offset_);
  }

  inline char *getDataByInternalId(tableint internal_id) const {
    return (data_level0_memory_ + internal_id * size_data_per_element_ +
            offsetData_);
  }

  int getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
  }

  size_t getMaxElements() { return max_elements_; }

  size_t getCurrentElementCount() { return cur_element_count; }

  size_t getDeletedCount() { return num_deleted_; }

  std::priority_queue<std::pair<dist_t, tableint>,
                      std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
  searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        candidateSet;

    dist_t lowerBound;
    if (!isMarkedDeleted(ep_id)) {
      dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id),
                                 dist_func_param_);
      top_candidates.emplace(dist, ep_id);
      lowerBound = dist;
      candidateSet.emplace(-dist, ep_id);
    } else {
      lowerBound = std::numeric_limits<dist_t>::max();
      candidateSet.emplace(-lowerBound, ep_id);
    }
    visited_array[ep_id] = visited_array_tag;

    while (!candidateSet.empty()) {
      std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
      if ((-curr_el_pair.first) > lowerBound &&
          top_candidates.size() == ef_construction_) {
        break;
      }
      candidateSet.pop();

      tableint curNodeNum = curr_el_pair.second;

      int *data; // = (int *)(linkList0_ + curNodeNum *
                 // size_links_per_element0_);
      if (layer == 0) {
        data = (int *)get_linklist0(curNodeNum);
      } else {
        data = (int *)get_linklist(curNodeNum, layer);
        //                    data = (int *) (linkLists_[curNodeNum] + (layer -
        //                    1) * size_links_per_element_);
      }
      size_t size = getListCount((linklistsizeint *)data);
      tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
      _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
      _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

      for (size_t j = 0; j < size; j++) {
        tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
        if (visited_array[candidate_id] == visited_array_tag)
          continue;
        visited_array[candidate_id] = visited_array_tag;
        char *currObj1 = (getDataByInternalId(candidate_id));

        dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
        if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
          candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
          _mm_prefetch(getDataByInternalId(candidateSet.top().second),
                       _MM_HINT_T0);
#endif

          if (!isMarkedDeleted(candidate_id))
            top_candidates.emplace(dist1, candidate_id);

          if (top_candidates.size() > ef_construction_)
            top_candidates.pop();

          if (!top_candidates.empty())
            lowerBound = top_candidates.top().first;
        }
      }
    }
    visited_list_pool_->releaseVisitedList(vl);

    return top_candidates;
  }

  template <bool has_deletions, bool collect_metrics = false>
  std::priority_queue<std::pair<dist_t, tableint>,
                      std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
  searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef,
                    BaseFilterFunctor *isIdAllowed = nullptr) const {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        candidate_set;

    dist_t lowerBound;
    if ((!has_deletions || !isMarkedDeleted(ep_id)) &&
        ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))) {
      dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id),
                                 dist_func_param_);
      lowerBound = dist;
      top_candidates.emplace(dist, ep_id);
      candidate_set.emplace(-dist, ep_id);
    } else {
      lowerBound = std::numeric_limits<dist_t>::max();
      candidate_set.emplace(-lowerBound, ep_id);
    }

    visited_array[ep_id] = visited_array_tag;

    while (!candidate_set.empty()) {
      std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

      if ((-current_node_pair.first) > lowerBound &&
          (top_candidates.size() == ef || (!isIdAllowed && !has_deletions))) {
        break;
      }
      candidate_set.pop();

      tableint current_node_id = current_node_pair.second;
      int *data = (int *)get_linklist0(current_node_id);
      size_t size = getListCount((linklistsizeint *)data);


#ifdef USE_SSE
      _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
      _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(data_level0_memory_ +
                       (*(data + 1)) * size_data_per_element_ + offsetData_,
                   _MM_HINT_T0);
      _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

      for (size_t j = 1; j <= size; j++) {
        int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
        _mm_prefetch(data_level0_memory_ +
                         (*(data + j + 1)) * size_data_per_element_ +
                         offsetData_,
                     _MM_HINT_T0); ////////////
#endif
        if (!(visited_array[candidate_id] == visited_array_tag)) {
          visited_array[candidate_id] = visited_array_tag;

          char *currObj1 = (getDataByInternalId(candidate_id));
          dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

          if (top_candidates.size() < ef || lowerBound > dist) {
            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
            _mm_prefetch(data_level0_memory_ +
                             candidate_set.top().second *
                                 size_data_per_element_ +
                             offsetLevel0_, ///////////
                         _MM_HINT_T0);      ////////////////////////
#endif

            if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
                ((!isIdAllowed) ||
                 (*isIdAllowed)(getExternalLabel(candidate_id))))
              top_candidates.emplace(dist, candidate_id);

            if (top_candidates.size() > ef)
              top_candidates.pop();

            if (!top_candidates.empty())
              lowerBound = top_candidates.top().first;
          }
        }
      }
    }

    visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
  }

  void getNeighborsByHeuristic2(
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates,
      const size_t M) {
    if (top_candidates.size() < M) {
      return;
    }

    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    std::vector<std::pair<dist_t, tableint>> return_list;
    while (top_candidates.size() > 0) {
      queue_closest.emplace(-top_candidates.top().first,
                            top_candidates.top().second);
      top_candidates.pop();
    }

    while (queue_closest.size()) {
      if (return_list.size() >= M)
        break;
      std::pair<dist_t, tableint> curent_pair = queue_closest.top();
      dist_t dist_to_query = -curent_pair.first;
      queue_closest.pop();
      bool good = true;

      for (std::pair<dist_t, tableint> second_pair : return_list) {
        dist_t curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                      getDataByInternalId(curent_pair.second),
                                      dist_func_param_);
        if (curdist < dist_to_query) {
          good = false;
          break;
        }
      }
      if (good) {
        return_list.push_back(curent_pair);
      }
    }

    for (std::pair<dist_t, tableint> curent_pair : return_list) {
      top_candidates.emplace(-curent_pair.first, curent_pair.second);
    }
  }

  linklistsizeint *get_linklist0(tableint internal_id) const {
    return (linklistsizeint *)(data_level0_memory_ +
                               internal_id * size_data_per_element_ +
                               offsetLevel0_);
  }

  linklistsizeint *get_linklist0(tableint internal_id,
                                 char *data_level0_memory_) const {
    return (linklistsizeint *)(data_level0_memory_ +
                               internal_id * size_data_per_element_ +
                               offsetLevel0_);
  }

  linklistsizeint *get_linklist(tableint internal_id, int level) const {
    return (linklistsizeint *)(linkLists_[internal_id] +
                               (level - 1) * size_links_per_element_);
  }

  linklistsizeint *get_linklist_at_level(tableint internal_id,
                                         int level) const {
    return level == 0 ? get_linklist0(internal_id)
                      : get_linklist(internal_id, level);
  }

  tableint mutuallyConnectNewElement(
      const void *, tableint cur_c,
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates,
      int level, bool isUpdate) {
    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(top_candidates, M_);

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(M_);
    while (top_candidates.size() > 0) {
      selectedNeighbors.push_back(top_candidates.top().second);
      top_candidates.pop();
    }

    tableint next_closest_entry_point = selectedNeighbors.back();

    {
      linklistsizeint *ll_cur;
      if (level == 0)
        ll_cur = get_linklist0(cur_c);
      else
        ll_cur = get_linklist(cur_c, level);

      setListCount(ll_cur, selectedNeighbors.size());
      tableint *data = (tableint *)(ll_cur + 1);
      for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

        data[idx] = selectedNeighbors[idx];
      }
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

      linklistsizeint *ll_other;
      if (level == 0)
        ll_other = get_linklist0(selectedNeighbors[idx]);
      else
        ll_other = get_linklist(selectedNeighbors[idx], level);

      size_t sz_link_list_other = getListCount(ll_other);

      tableint *data = (tableint *)(ll_other + 1);

      bool is_cur_c_present = false;
      if (isUpdate) {
        for (size_t j = 0; j < sz_link_list_other; j++) {
          if (data[j] == cur_c) {
            is_cur_c_present = true;
            break;
          }
        }
      }

      // If cur_c is already present in the neighboring connections of
      // `selectedNeighbors[idx]` then no need to modify any connections or run
      // the heuristics.
      if (!is_cur_c_present) {
        if (sz_link_list_other < Mcurmax) {
          data[sz_link_list_other] = cur_c;
          setListCount(ll_other, sz_link_list_other + 1);
        } else {
          // finding the "weakest" element to replace it with the new one
          dist_t d_max = fstdistfunc_(
              getDataByInternalId(cur_c),
              getDataByInternalId(selectedNeighbors[idx]), dist_func_param_);
          // Heuristic:
          std::priority_queue<std::pair<dist_t, tableint>,
                              std::vector<std::pair<dist_t, tableint>>,
                              CompareByFirst>
              candidates;
          candidates.emplace(d_max, cur_c);

          for (size_t j = 0; j < sz_link_list_other; j++) {
            candidates.emplace(
                fstdistfunc_(getDataByInternalId(data[j]),
                             getDataByInternalId(selectedNeighbors[idx]),
                             dist_func_param_),
                data[j]);
          }

          getNeighborsByHeuristic2(candidates, Mcurmax);

          int indx = 0;
          while (candidates.size() > 0) {
            data[indx] = candidates.top().second;
            candidates.pop();
            indx++;
          }

          setListCount(ll_other, indx);
          // Nearest K:
          /*int indx = -1;
          for (int j = 0; j < sz_link_list_other; j++) {
              dist_t d = fstdistfunc_(getDataByInternalId(data[j]),
          getDataByInternalId(rez[idx]), dist_func_param_); if (d > d_max) {
                  indx = j;
                  d_max = d;
              }
          }
          if (indx >= 0) {
              data[indx] = cur_c;
          } */
        }
      }
    }

    return next_closest_entry_point;
  }

  void resizeIndex(size_t new_max_elements) {

    delete visited_list_pool_;
    visited_list_pool_ = new VisitedListPool(1, new_max_elements);

    element_levels_.resize(new_max_elements);

    // Reallocate base layer
    char *data_level0_memory_new = (char *)realloc(
        data_level0_memory_, new_max_elements * size_data_per_element_);
    data_level0_memory_ = data_level0_memory_new;

    // Reallocate all other layers
    char **linkLists_new =
        (char **)realloc(linkLists_, sizeof(void *) * new_max_elements);
    linkLists_ = linkLists_new;

    max_elements_ = new_max_elements;
  }

  void saveIndex(const std::string &location) {
    std::ofstream output(location, std::ios::binary);
    std::streampos position;

    writeBinaryPOD(output, offsetLevel0_);
    writeBinaryPOD(output, max_elements_);
    writeBinaryPOD(output, cur_element_count);
    writeBinaryPOD(output, size_data_per_element_);
    writeBinaryPOD(output, label_offset_);
    writeBinaryPOD(output, offsetData_);
    writeBinaryPOD(output, maxlevel_);
    writeBinaryPOD(output, enterpoint_node_);
    writeBinaryPOD(output, maxM_);

    writeBinaryPOD(output, maxM0_);
    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, mult_);
    writeBinaryPOD(output, ef_construction_);

    output.write(data_level0_memory_,
                 cur_element_count * size_data_per_element_);

    for (size_t i = 0; i < cur_element_count; i++) {
      unsigned int linkListSize =
          element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i]
                                 : 0;
      writeBinaryPOD(output, linkListSize);
      if (linkListSize)
        output.write(linkLists_[i], linkListSize);
    }
    output.close();
  }

  void loadIndex(const std::string &location, SpaceInterface<dist_t> *s,
                 size_t max_elements_i = 0) {
    std::ifstream input(location, std::ios::binary);

    // get file size:
    input.seekg(0, input.end);
    std::streampos total_filesize = input.tellg();
    input.seekg(0, input.beg);

    readBinaryPOD(input, offsetLevel0_);
    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, cur_element_count);

    size_t max_elements = max_elements_i;
    if (max_elements < cur_element_count)
      max_elements = max_elements_;
    max_elements_ = max_elements;
    readBinaryPOD(input, size_data_per_element_);
    readBinaryPOD(input, label_offset_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, enterpoint_node_);

    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, mult_);
    readBinaryPOD(input, ef_construction_);

    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();

    auto pos = input.tellg();

    /// Optional - check if index is ok:
    input.seekg(cur_element_count * size_data_per_element_, input.cur);
    for (size_t i = 0; i < cur_element_count; i++) {

      unsigned int linkListSize;
      readBinaryPOD(input, linkListSize);
      if (linkListSize != 0) {
        input.seekg(linkListSize, input.cur);
      }
    }

    input.clear();
    /// Optional check end

    input.seekg(pos, input.beg);

    data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    size_links_per_element_ =
        maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

    visited_list_pool_ = new VisitedListPool(1, max_elements);

    linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
    element_levels_ = std::vector<int>(max_elements);
    revSize_ = 1.0 / mult_;
    ef_ = 10;
    for (size_t i = 0; i < cur_element_count; i++) {
      label_lookup_[getExternalLabel(i)] = i;
      unsigned int linkListSize;
      readBinaryPOD(input, linkListSize);
      if (linkListSize == 0) {
        element_levels_[i] = 0;
        linkLists_[i] = nullptr;
      } else {
        element_levels_[i] = linkListSize / size_links_per_element_;
        linkLists_[i] = (char *)malloc(linkListSize);
        input.read(linkLists_[i], linkListSize);
      }
    }

    for (size_t i = 0; i < cur_element_count; i++) {
      if (isMarkedDeleted(i)) {
        num_deleted_ += 1;
        if (allow_replace_deleted_)
          deleted_elements.insert(i);
      }
    }

    input.close();

    return;
  }

  template <typename data_t>
  std::vector<data_t> getDataByLabel(labeltype label) const {
    auto search = label_lookup_.find(label);
    tableint internalId = search->second;

    char *data_ptrv = getDataByInternalId(internalId);
    size_t dim = *((size_t *)dist_func_param_);
    std::vector<data_t> data;
    data_t *data_ptr = (data_t *)data_ptrv;
    for (int i = 0; i < (int)dim; i++) {
      data.push_back(*data_ptr);
      data_ptr += 1;
    }
    return data;
  }

  /*
   * Marks an element with the given label deleted, does NOT really change the
   * current graph.
   */
  void markDelete(labeltype label) {
    auto search = label_lookup_.find(label);
    tableint internalId = search->second;

    markDeletedInternal(internalId);
  }

  /*
   * Uses the last 16 bits of the memory for the linked list size to store the
   * mark, whereas maxM0_ has to be limited to the lower 16 bits, however, still
   * large enough in almost all cases.
   */
  void markDeletedInternal(tableint internalId) {
    assert(internalId < cur_element_count);
    if (!isMarkedDeleted(internalId)) {
      unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
      *ll_cur |= DELETE_MARK;
      num_deleted_ += 1;
      if (allow_replace_deleted_) {
        deleted_elements.insert(internalId);
      }
    }
  }

  /*
   * Removes the deleted mark of the node, does NOT really change the current
   * graph.
   *
   * Note: the method is not safe to use when replacement of deleted elements is
   * enabled, because elements marked as deleted can be completely removed by
   * addPoint
   */
  void unmarkDelete(labeltype label) {
    auto search = label_lookup_.find(label);
    tableint internalId = search->second;

    unmarkDeletedInternal(internalId);
  }

  /*
   * Remove the deleted mark of the node.
   */
  void unmarkDeletedInternal(tableint internalId) {
    assert(internalId < cur_element_count);
    if (isMarkedDeleted(internalId)) {
      unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
      *ll_cur &= ~DELETE_MARK;
      num_deleted_ -= 1;
      if (allow_replace_deleted_) {
        deleted_elements.erase(internalId);
      }
    }
  }

  /*
   * Checks the first 16 bits of the memory to see if the element is marked
   * deleted.
   */
  bool isMarkedDeleted(tableint internalId) const {
    unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
    return *ll_cur & DELETE_MARK;
  }

  unsigned short int getListCount(linklistsizeint *ptr) const {
    return *((unsigned short int *)ptr);
  }

  void setListCount(linklistsizeint *ptr, unsigned short int size) const {
    *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
  }

  /*
   * Adds point. Updates the point if it is already in the index.
   * If replacement of deleted elements is enabled: replaces previously deleted
   * point if any, updating it with new point
   */
  void addPoint(const void *data_point, labeltype label,
                bool replace_deleted = false) {

    if (!replace_deleted) {
      addPoint(data_point, label, -1);
      return;
    }
    // check if there is vacant place
    tableint internal_id_replaced;
    bool is_vacant_place = !deleted_elements.empty();
    if (is_vacant_place) {
      internal_id_replaced = *deleted_elements.begin();
      deleted_elements.erase(internal_id_replaced);
    }

    // if there is no vacant place then add or update point
    // else add point to vacant place
    if (!is_vacant_place) {
      addPoint(data_point, label, -1);
    } else {
      // we assume that there are no concurrent operations on deleted element
      labeltype label_replaced = getExternalLabel(internal_id_replaced);
      setExternalLabel(internal_id_replaced, label);

      label_lookup_.erase(label_replaced);
      label_lookup_[label] = internal_id_replaced;

      unmarkDeletedInternal(internal_id_replaced);
      updatePoint(data_point, internal_id_replaced, 1.0);
    }
  }

  void updatePoint(const void *dataPoint, tableint internalId,
                   float updateNeighborProbability) {
    // update the feature vector associated with existing point with new vector
    memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

    int maxLevelCopy = maxlevel_;
    tableint entryPointCopy = enterpoint_node_;
    // If point to be updated is entry point and graph just contains single
    // element then just return.
    if (entryPointCopy == internalId && cur_element_count == 1)
      return;

    int elemLevel = element_levels_[internalId];
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int layer = 0; layer <= elemLevel; layer++) {
      std::unordered_set<tableint> sCand;
      std::unordered_set<tableint> sNeigh;
      std::vector<tableint> listOneHop =
          getConnectionsWithLock(internalId, layer);
      if (listOneHop.size() == 0)
        continue;

      sCand.insert(internalId);

      for (auto &&elOneHop : listOneHop) {
        sCand.insert(elOneHop);

        if (distribution(update_probability_generator_) >
            updateNeighborProbability)
          continue;

        sNeigh.insert(elOneHop);

        std::vector<tableint> listTwoHop =
            getConnectionsWithLock(elOneHop, layer);
        for (auto &&elTwoHop : listTwoHop) {
          sCand.insert(elTwoHop);
        }
      }

      for (auto &&neigh : sNeigh) {
        // if (neigh == internalId)
        //     continue;

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            candidates;
        size_t size =
            sCand.find(neigh) == sCand.end()
                ? sCand.size()
                : sCand.size() - 1; // sCand guaranteed to have size >= 1
        size_t elementsToKeep = std::min(ef_construction_, size);
        for (auto &&cand : sCand) {
          if (cand == neigh)
            continue;

          dist_t distance =
              fstdistfunc_(getDataByInternalId(neigh),
                           getDataByInternalId(cand), dist_func_param_);
          if (candidates.size() < elementsToKeep) {
            candidates.emplace(distance, cand);
          } else {
            if (distance < candidates.top().first) {
              candidates.pop();
              candidates.emplace(distance, cand);
            }
          }
        }

        // Retrieve neighbours using heuristic and set connections.
        getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

        {
          linklistsizeint *ll_cur;
          ll_cur = get_linklist_at_level(neigh, layer);
          size_t candSize = candidates.size();
          setListCount(ll_cur, candSize);
          tableint *data = (tableint *)(ll_cur + 1);
          for (size_t idx = 0; idx < candSize; idx++) {
            data[idx] = candidates.top().second;
            candidates.pop();
          }
        }
      }
    }

    repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel,
                               maxLevelCopy);
  }

  void repairConnectionsForUpdate(const void *dataPoint,
                                  tableint entryPointInternalId,
                                  tableint dataPointInternalId,
                                  int dataPointLevel, int maxLevel) {
    tableint currObj = entryPointInternalId;
    if (dataPointLevel < maxLevel) {
      dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj),
                                    dist_func_param_);
      for (int level = maxLevel; level > dataPointLevel; level--) {
        bool changed = true;
        while (changed) {
          changed = false;
          unsigned int *data;
          data = get_linklist_at_level(currObj, level);
          int size = getListCount(data);
          tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
          _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
          for (int i = 0; i < size; i++) {
#ifdef USE_SSE
            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
            tableint cand = datal[i];
            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand),
                                    dist_func_param_);
            if (d < curdist) {
              curdist = d;
              currObj = cand;
              changed = true;
            }
          }
        }
      }
    }

    for (int level = dataPointLevel; level >= 0; level--) {
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst>
          topCandidates = searchBaseLayer(currObj, dataPoint, level);

      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst>
          filteredTopCandidates;
      while (topCandidates.size() > 0) {
        if (topCandidates.top().second != dataPointInternalId)
          filteredTopCandidates.push(topCandidates.top());

        topCandidates.pop();
      }

      // Since element_levels_ is being used to get `dataPointLevel`, there
      // could be cases where `topCandidates` could just contains entry point
      // itself. To prevent self loops, the `topCandidates` is filtered and thus
      // can be empty.
      if (filteredTopCandidates.size() > 0) {
        bool epDeleted = isMarkedDeleted(entryPointInternalId);
        if (epDeleted) {
          filteredTopCandidates.emplace(
              fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId),
                           dist_func_param_),
              entryPointInternalId);
          if (filteredTopCandidates.size() > ef_construction_)
            filteredTopCandidates.pop();
        }

        currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId,
                                            filteredTopCandidates, level, true);
      }
    }
  }

  std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
    unsigned int *data = get_linklist_at_level(internalId, level);
    int size = getListCount(data);
    std::vector<tableint> result(size);
    tableint *ll = (tableint *)(data + 1);
    memcpy(result.data(), ll, size * sizeof(tableint));
    return result;
  }

  tableint addPoint(const void *data_point, labeltype label, int level) {
    tableint cur_c = 0;
    {
      // Checking if the element with the same label already exists
      // if so, updating it *instead* of creating a new element.
      auto search = label_lookup_.find(label);
      if (search != label_lookup_.end()) {
        tableint existingInternalId = search->second;

        if (isMarkedDeleted(existingInternalId)) {
          unmarkDeletedInternal(existingInternalId);
        }
        updatePoint(data_point, existingInternalId, 1.0);

        return existingInternalId;
      }

      cur_c = cur_element_count;
      cur_element_count++;
      label_lookup_[label] = cur_c;
    }

    int curlevel = getRandomLevel(mult_);
    if (level > 0)
      curlevel = level;

    element_levels_[cur_c] = curlevel;

    int maxlevelcopy = maxlevel_;
    tableint currObj = enterpoint_node_;
    tableint enterpoint_copy = enterpoint_node_;

    memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_,
           0, size_data_per_element_);

    // Initialisation of the data and label
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);

    if (curlevel) {
      linkLists_[cur_c] =
          (char *)malloc(size_links_per_element_ * curlevel + 1);
      memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
    }

    if ((signed)currObj != -1) {
      if (curlevel < maxlevelcopy) {
        dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj),
                                      dist_func_param_);
        for (int level = maxlevelcopy; level > curlevel; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int *data;
            data = get_linklist(currObj, level);
            int size = getListCount(data);

            tableint *datal = (tableint *)(data + 1);
            for (int i = 0; i < size; i++) {
              tableint cand = datal[i];
              dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand),
                                      dist_func_param_);
              if (d < curdist) {
                curdist = d;
                currObj = cand;
                changed = true;
              }
            }
          }
        }
      }

      bool epDeleted = isMarkedDeleted(enterpoint_copy);
      for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates = searchBaseLayer(currObj, data_point, level);
        if (epDeleted) {
          top_candidates.emplace(
              fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy),
                           dist_func_param_),
              enterpoint_copy);
          if (top_candidates.size() > ef_construction_)
            top_candidates.pop();
        }
        currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates,
                                            level, false);
      }
    } else {
      // Do nothing for the first element
      enterpoint_node_ = 0;
      maxlevel_ = curlevel;
    }

    // Releasing lock for the maximum level
    if (curlevel > maxlevelcopy) {
      enterpoint_node_ = cur_c;
      maxlevel_ = curlevel;
    }
    return cur_c;
  }

  std::priority_queue<std::pair<dist_t, labeltype>>
  searchKnn(const void *query_data, size_t k,
            BaseFilterFunctor *isIdAllowed = nullptr) const {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count == 0)
      return result;

    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(
        query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

    for (int level = maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;
        unsigned int *data;

        data = (unsigned int *)get_linklist(currObj, level);
        int size = getListCount(data);

        tableint *datal = (tableint *)(data + 1);
        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand),
                                  dist_func_param_);

          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;
    if (num_deleted_) {
      top_candidates = searchBaseLayerST<true, true>(
          currObj, query_data, std::max(ef_, k), isIdAllowed);
    } else {
      top_candidates = searchBaseLayerST<false, true>(
          currObj, query_data, std::max(ef_, k), isIdAllowed);
    }

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
      std::pair<dist_t, tableint> rez = top_candidates.top();
      result.push(std::pair<dist_t, labeltype>(rez.first,
                                               getExternalLabel(rez.second)));
      top_candidates.pop();
    }
    return result;
  }

  void checkIntegrity() {
    int connections_checked = 0;
    std::vector<int> inbound_connections_num(cur_element_count, 0);
    for (int i = 0; i < cur_element_count; i++) {
      for (int l = 0; l <= element_levels_[i]; l++) {
        linklistsizeint *ll_cur = get_linklist_at_level(i, l);
        int size = getListCount(ll_cur);
        tableint *data = (tableint *)(ll_cur + 1);
        std::unordered_set<tableint> s;
        for (int j = 0; j < size; j++) {
          assert(data[j] > 0);
          assert(data[j] < cur_element_count);
          assert(data[j] != i);
          inbound_connections_num[data[j]]++;
          s.insert(data[j]);
          connections_checked++;
        }
        assert(s.size() == size);
      }
    }
    if (cur_element_count > 1) {
      int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
      for (int i = 0; i < cur_element_count; i++) {
        assert(inbound_connections_num[i] > 0);
        min1 = std::min(inbound_connections_num[i], min1);
        max1 = std::max(inbound_connections_num[i], max1);
      }
      std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
    }
    std::cout << "integrity ok, checked " << connections_checked
              << " connections\n";
  }
};
} // namespace hnswlib
/* =================================================================================================== */
/*                                        hnswlib/space_l2.h                                           */
namespace hnswlib {

static float
L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

#if defined(USE_AVX512)

// Favor using AVX512 if available.
static float
L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
            TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
            TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}
#endif

#if defined(USE_AVX)

// Favor using AVX if available.
static float
L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

#endif

#if defined(USE_SSE)

static float
L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC<float> L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;

static float
L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}
#endif


#if defined(USE_SSE)
static float
L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);


    size_t qty4 = qty >> 2;

    const float *pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float
L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}
#endif

class L2Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2Space(size_t dim) {
        fstdistfunc_ = L2Sqr;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
        else if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #elif defined(USE_AVX)
        if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = L2SqrSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = L2SqrSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = L2SqrSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2Space() {}
};

static int
L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    int res = 0;
    unsigned char *a = (unsigned char *) pVect1;
    unsigned char *b = (unsigned char *) pVect2;

    qty = qty >> 2;
    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}

static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    int res = 0;
    unsigned char* a = (unsigned char*)pVect1;
    unsigned char* b = (unsigned char*)pVect2;

    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}

class L2SpaceI : public SpaceInterface<int> {
    DISTFUNC<int> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2SpaceI(size_t dim) {
        if (dim % 4 == 0) {
            fstdistfunc_ = L2SqrI4x;
        } else {
            fstdistfunc_ = L2SqrI;
        }
        dim_ = dim;
        data_size_ = dim * sizeof(unsigned char);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<int> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2SpaceI() {}
};
}  // namespace hnswlib
/* =================================================================================================== */
/*                                        hnswlib/space_ip.h                                           */
namespace hnswlib {

static float
InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float *) pVect1)[i] * ((float *) pVect2)[i];
    }
    return res;
}

static float
InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
}

#if defined(USE_AVX)

// Favor using AVX if available.
static float
InnerProductSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float *pEnd1 = pVect1 + 16 * qty16;
    const float *pEnd2 = pVect1 + 4 * qty4;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    __m128 v1, v2;
    __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    return sum;
}

static float
InnerProductDistanceSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE)

static float
InnerProductSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float *pEnd1 = pVect1 + 16 * qty16;
    const float *pEnd2 = pVect1 + 4 * qty4;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

static float
InnerProductDistanceSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif


#if defined(USE_AVX512)

static float
InnerProductSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;


    const float *pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m512 v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
    }

    _mm512_store_ps(TmpRes, sum512);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] + TmpRes[15];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_AVX)

static float
InnerProductSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;


    const float *pEnd1 = pVect1 + 16 * qty16;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    _mm256_store_ps(TmpRes, sum256);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE)

static float
InnerProductSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;

    const float *pEnd1 = pVect1 + 16 * qty16;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC<float> InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
static DISTFUNC<float> InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
static DISTFUNC<float> InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
static DISTFUNC<float> InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;

static float
InnerProductDistanceSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

static float
InnerProductDistanceSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}
#endif

class InnerProductSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    InnerProductSpace(size_t dim) {
        fstdistfunc_ = InnerProductDistance;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
        } else if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #elif defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #endif
    #if defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
            InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
        }
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

~InnerProductSpace() {}
};

}  // namespace hnswlib
/* =================================================================================================== */
/*                                          hnsw/hnsw.hpp                                              */
#include <chrono>
#include <memory>

namespace hnsw {

struct HNSW : public Builder {
  int nb;                 // 数据量
  int dim;                // 数据维度
  int M;                  // 每次插入节点到 HNSW 时，在每一层搜索出了该节点的最近邻居，从这些邻居中选出要和该节点进行双向连接的点的个数
  int efConstruction;     // 每次插入节点到 HNSW 时，在每一层搜索该节点的最近邻居的个数
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw = nullptr;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space = nullptr;                    // 距离度量空间

  Graph<int> final_graph;           // 最终构建出来的有向图

  HNSW(int dim, const std::string &metric, int R = 32, int L = 200)
      : dim(dim), M(R / 2), efConstruction(L) {
    auto m = metric_map[metric];
    if (m == Metric::L2) {
      space = std::make_unique<hnswlib::L2Space>(dim);
    } else if (m == Metric::IP) {
      space = std::make_unique<hnswlib::InnerProductSpace>(dim);
    } else {
#ifdef DEBUG_ENABLED
      fprintf(stderr, "Unsupported metric type\n");
#endif
    }
  }

  void Build(float *data, int N) override {
    nb = N;
    hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), N, M, efConstruction);
#ifdef DEBUG_ENABLED
    int cnt = 0;
    auto st = std::chrono::high_resolution_clock::now();
#endif
    for (int i = 0; i < nb; ++i) {
      hnsw->addPoint(data + i * dim, i);
#ifdef DEBUG_ENABLED
      int cur = cnt += 1;
      if (cur % 10000 == 0) {
        fprintf(stderr, "HNSW building progress: [%d/%d]\n", cur, nb);
      }
#endif
    }
#ifdef DEBUG_ENABLED
    auto ed = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(ed - st).count();
    fprintf(stderr, "HNSW building cost: %.2lfs\n", ela);
#endif
    // 把 hnsw 的 level = 0 的 layer 复制给 final_graph
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
/* =================================================================================================== */
/* =================================================================================================== */
/* =================================================================================================== */

#define K_MAX 10
#define K_MIN 3

// Data Part
int N, D;
float *database;

// ==========================================================================================

std::unique_ptr<hnsw::SearcherBase> searcher;

// Preprocess Part
void preprocess() {
    int R = 32;
    int L = 200;
    std::unique_ptr<hnsw::Builder> index = std::unique_ptr<hnsw::Builder>((hnsw::Builder *)new hnsw::HNSW(D, "L2", R, L));
#ifdef DEBUG_ENABLED
    fprintf(stderr, "=============HNSW parameters=============\n");
    fprintf(stderr, "M: %d\n", R);
    fprintf(stderr, "efConstruction: %d\n", L);
    fprintf(stderr, "=========================================\n");
#endif
    index->Build(database, N);
    hnsw::Graph<int> graph = index->GetGraph();
    int level = 0;
    int ef = 50000;
    searcher = create_searcher(graph, "L2", level);
    searcher->SetData(database, N, D);
    searcher->SetEf(ef);
#ifdef DEBUG_ENABLED
    fprintf(stderr, "=============Search parameters=============\n");
    fprintf(stderr, "optimize level: %d\n", level);
    fprintf(stderr, "ef: %d\n", ef);
    fprintf(stderr, "===========================================\n");
#endif
}

// ==========================================================================================


// ==========================================================================================

// Query Part
void query(const float *q, const int k, int *idxs) {
    searcher->Search(q, k, idxs);
}

// ==========================================================================================
int main() {
#ifdef DEBUG_ENABLED
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
    scanf("%d", &N);
    scanf("%d", &D);
    database = new float[N * D];
    for (int i = 0; i < (N * D); i++) {
      scanf("%f", (database + i));
    }
#ifdef DEBUG_ENABLED
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    double ela = std::chrono::duration<double>(t2 - t1).count();
    fprintf(stderr, "ReadIO cost: %.4lfs\n", ela);
#endif

#ifdef DEBUG_ENABLED
    t1 = std::chrono::high_resolution_clock::now();
#endif
    preprocess();
#ifdef DEBUG_ENABLED
    t2 = std::chrono::high_resolution_clock::now();
    ela = std::chrono::duration<double>(t2 - t1).count();
    fprintf(stderr, "Preprocess cost: %.4lfs\n", ela);
#endif

    float *q = new float[D];

    printf("ok\n");
    fflush(stdout);

    int k;
    scanf("%d", &k);

    int *idxs = new int[k];

#ifdef DEBUG_ENABLED
    t1 = std::chrono::high_resolution_clock::now();
#endif
    while (true) {
#ifdef DEBUG_ENABLED
    auto read_t1 = std::chrono::high_resolution_clock::now();
#endif
        for (int i = 0; i < D; ++i) {
            if (scanf("%f", (q + i)) == 0) goto out;
        }
#ifdef DEBUG_ENABLED
    auto read_t2 = std::chrono::high_resolution_clock::now();
    auto read_ela = std::chrono::duration<double>(read_t2 - read_t1).count();
    fprintf(stderr, "Read query cost: %.4lfs\n", read_ela);
#endif
#ifdef DEBUG_ENABLED
    auto query_t1 = std::chrono::high_resolution_clock::now();
#endif
        query(q, k, idxs);
#ifdef DEBUG_ENABLED
    auto query_t2 = std::chrono::high_resolution_clock::now();
    auto query_ela = std::chrono::duration<double>(query_t2 - query_t1).count();
    fprintf(stderr, "Query cost: %.4lfs\n", query_ela);
#endif
#ifdef DEBUG_ENABLED
    auto write_t1 = std::chrono::high_resolution_clock::now();
#endif
        for (int i = 0; i < k; i++) {
            printf("%d ", idxs[i]);
        }
        printf("\n");
        fflush(stdout);

#ifdef DEBUG_ENABLED
    auto write_t2 = std::chrono::high_resolution_clock::now();
    auto write_ela = std::chrono::duration<double>(write_t2 - write_t1).count();
    fprintf(stderr, "Write result cost: %.4lfs\n", write_ela);
#endif
    }
out:
#ifdef DEBUG_ENABLED
    t2 = std::chrono::high_resolution_clock::now();
    ela = std::chrono::duration<double>(t2 - t1).count();
    fprintf(stderr, "Totoal query cost: %.4lfs\n", ela);
#endif
    return 0;
}