#include <stdio.h>

#include <hnsw/hnswlib/hnswlib.h>

#define DEBUG_ENABLED

#define K_MAX 10
#define K_MIN 3

typedef unsigned int docidtype;
typedef float dist_t;

// Data Part
int N, D;
float *database;

// ==========================================================================================

hnswlib::L2Space *space;
hnswlib::HierarchicalNSW<float>* alg_hnsw;

// Preprocess Part
void preprocess() {
    int M = 32;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    space = new hnswlib::L2Space(D);
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(space, N, M, ef_construction);

    // Add data to index
    for (int i = 0; i < N; i++) {
        alg_hnsw->addPoint(database + i * D, i);
    }

#ifdef DEBUG_ENABLED
    fprintf(stderr, "=============HNSW parameters=============\n");
    fprintf(stderr, "M: %d\n", M);
    fprintf(stderr, "ef_construction: %d\n", ef_construction);
    fprintf(stderr, "=========================================\n");
#endif

    size_t ef = 1024;
    alg_hnsw->setEf(ef);

#ifdef DEBUG_ENABLED
    fprintf(stderr, "=============search parameters=============\n");
    fprintf(stderr, "ef: %d\n", ef);
    fprintf(stderr, "===========================================\n");
#endif
}

// ==========================================================================================


// ==========================================================================================

// Query Part
void query(const float *q, const int k, int *idxs) {
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(q, k);
    for (int i = 0; i < k; i++) {
        idxs[i] = result.top().second;
        result.pop();
    }
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
        // Output k nearest indices
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
    delete space;
    delete alg_hnsw; 
    return 0;
}