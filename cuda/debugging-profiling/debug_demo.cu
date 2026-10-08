#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr int N = 20;
constexpr int BLOCK = 8;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                     __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE);                                  \
    }                                                              \
} while (0)

__global__ void vector_add(const int *a, const int *b, int *c) {
#ifdef DEMO_BUG
    int i = blockIdx.x * blockDim.x + threadIdx.x + BLOCK; // deliberate error
#else
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int a[N], b[N], c[N];
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = 2 * i;
        c[i] = -1; // makes untouched elements visible
    }

    int *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&da), sizeof(a)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&db), sizeof(b)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dc), sizeof(c)));
    CUDA_CHECK(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, c, sizeof(c), cudaMemcpyHostToDevice));

    const int grid = (N + BLOCK - 1) / BLOCK;
    vector_add<<<grid, BLOCK>>>(da, db, dc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c, dc, sizeof(c), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        const int expected = a[i] + b[i];
        if (c[i] != expected) ++mismatches;
        std::printf("i=%2d  expected=%2d  actual=%2d  %s\n", i,
                    expected, c[i], c[i] == expected ? "OK" : "WRONG");
    }
    std::printf("Verification: %s (%d mismatches)\n",
                mismatches == 0 ? "PASS" : "FAIL", mismatches);

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
