#include "amx-gemm.h"

#define A_PAD 64
#define B_PAD 16
#define C_PAD 16

#define TM 512
#define TN 512
#define TK 512

#ifndef CORES_M
#define CORES_M 4
#endif

#ifndef CORES_N
#define CORES_N 4
#endif

#ifndef CORES
#define CORES (CORES_M * CORES_N)
#endif

#ifndef IS_TEST_MNK
#define IS_TEST_MNK 1
#endif

#ifndef IS_TEST_K
#define IS_TEST_K 0
#endif

#ifndef TEST_M
#define TEST_M 2048
#endif

#ifndef TEST_N
#define TEST_N 2048
#endif

// input:
//     A: [M, K] array
//     B: [K/KPACK, N*KPACK] array, where KPACK = (4/sizeof(type_t))
//     lda = K + PAD, ldb = N + PAD, ldc = N + PAD
// output:
//     C: [M, N] array
void cpu_gemm_i8i8i32(
        int8_t * __restrict__ A, int8_t * __restrict__ B,
        int32_t * __restrict__ C,
        const int M, const int N, const int K,
        const int lda, const int ldb, const int ldc) {

    assert(M > 0 && N > 0 && K > 0);
    assert(lda >= K && ldb >= N && ldc >= N);

    const int KPACK = KPACK_b8;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = C[OFFSET2D(i, j, ldc)];
            for (int k = 0; k < K; k++) {
                sum += A[OFFSET2D(i, k, lda)] *
                       B[OFFSET3D(k / KPACK, j, k % KPACK, ldb, KPACK)];
            }
            C[OFFSET2D(i, j, N)] = sum;
        }
    }
}

void amx_gemm_i8i8i32_naive(
        int8_t * __restrict__ A, int8_t * __restrict__ B,
        int32_t * __restrict__ C,
        const int M, const int N, const int K,
        const int lda, const int ldb, const int ldc) {

    assert(M > 0 && N > 0 && K > 0);
    assert(lda >= K && ldb >= N && ldc >= N);

    for (int i = 0; i < M; i += MAX_ROWS) {
        for (int j = 0; j < N; j += MAX_ROWS) {
            amx_tile_load_L2C(0, C, i, j, ldc);
            for (int k = 0; k < K; k += MAX_COLS) {
                amx_tile_load_L1A(1, A, i, k, lda);
                amx_tile_load_L1B(2, B, k, j, ldb);
                _tile_dpbssd(0, 1, 2);
            }
            amx_tile_store_L1C(0, C, i, j, ldc);
        }
    }
}

void amx_gemm_i8i8i32_l0_tiling_2A2B(
        int8_t * __restrict__ A, int8_t * __restrict__ B,
        int32_t * __restrict__ C,
        const int M, const int N, const int K,
        const int lda, const int ldb, const int ldc);

void amx_gemm_i8i8i32_l2_tiling(
        int8_t * __restrict__ A, int8_t * __restrict__ B,
        int32_t * __restrict__ C,
        const int M, const int N, const int K,
        const int lda, const int ldb, const int ldc);

void amx_gemm_i8i8i32_multi_thread(
        int8_t * __restrict__ A, int8_t * __restrict__ B,
        int32_t * __restrict__ C,
        const int M, const int N, const int K,
        const int lda, const int ldb, const int ldc);

void amx_gemm_i8i8i32(
        int8_t * __restrict__ A, int8_t * __restrict__ B,
        int32_t * __restrict__ C,
        const int M, const int N, const int K,
        const int lda, const int ldb, const int ldc) {

    assert(M > 0 && N > 0 && K > 0);
    assert(lda >= K && ldb >= N && ldc >= N);

    amx_gemm_i8i8i32_multi_thread(A, B, C, M, N, K, lda, ldb, ldc);
}

void amx_gemm_i8i8i32_l0_tiling_2A2B(
        int8_t * __restrict__ A, int8_t * __restrict__ B,
        int32_t * __restrict__ C,
        const int M, const int N, const int K,
        const int lda, const int ldb, const int ldc) {

    for (int i = 0; i < M; i += MAX_ROWS * 2) {
        for (int j = 0; j < N; j += MAX_ROWS * 2) {
            amx_tile_load_L2C(0, C, i, j, ldc);
            amx_tile_load_L2C(1, C, i, j + MAX_ROWS, ldc);
            amx_tile_load_L2C(2, C, i + MAX_ROWS, j, ldc);
            amx_tile_load_L2C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);
            #pragma unroll 32
            for (int k = 0; k < K; k += MAX_COLS) {
                amx_tile_load_L2B(6, B, k, j, ldb);
                amx_tile_load_L1A(4, A, i, k, lda);
                amx_tile_load_L2B(7, B, k, j + MAX_ROWS, ldb);
                _tile_dpbssd(0, 4, 6);
                amx_tile_load_L1A(5, A, i + MAX_ROWS, k, lda);
                _tile_dpbssd(1, 4, 7);
                _tile_dpbssd(2, 5, 6);
                _tile_dpbssd(3, 5, 7);
            }
            amx_tile_store_L1C(0, C, i, j, ldc);
            amx_tile_store_L1C(1, C, i, j + MAX_ROWS, ldc);
            amx_tile_store_L1C(2, C, i + MAX_ROWS, j, ldc);
            amx_tile_store_L1C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);
        }
    }
}

void amx_gemm_i8i8i32_l2_tiling(
        int8_t * __restrict__ A, int8_t * __restrict__ B,
        int32_t * __restrict__ C,
        const int M, const int N, const int K,
        const int lda, const int ldb, const int ldc) {

    for (int tm = 0; tm < M; tm += TM) {
        for (int tn = 0; tn < N; tn += TN) {
            for (int tk = 0; tk < K; tk += TK) {
                amx_gemm_i8i8i32_l0_tiling_2A2B(
                    &A[OFFSET2D(tm, tk, lda)],
                    &B[OFFSET2D(tk / KPACK_b8, tn * KPACK_b8, ldb * KPACK_b8)],
                    &C[OFFSET2D(tm, tn, ldc)],
                    (M - tm) < TM ? (M - tm) : TM,
                    (N - tn) < TN ? (N - tn) : TN,
                    (K - tk) < TK ? (K - tk) : TK,
                    lda, ldb, ldc);
            }
        }
    }
}

void amx_gemm_i8i8i32_multi_thread(
        int8_t * __restrict__ A, int8_t * __restrict__ B,
        int32_t * __restrict__ C,
        const int M, const int N, const int K,
        const int lda, const int ldb, const int ldc) {

    #pragma omp parallel num_threads(CORES)
    {
        int thread_id = omp_get_thread_num();
        bind_thread_to_cpu(thread_id);

        int tm = thread_id / CORES_N * TM;
        int tn = thread_id % CORES_N * TN;

        for (int i = tm; i < M; i += TM * CORES_M) {
            for (int j = tn; j < N; j += TN * CORES_N) {
                amx_gemm_i8i8i32_l2_tiling(
                    &A[OFFSET2D(i, 0, lda)],
                    &B[OFFSET2D(0, j * KPACK_b8, ldb)],
                    &C[OFFSET2D(i, j, ldc)],
                    (M - i) < TM ? (M - i) : TM,
                    (N - j) < TN ? (N - j) : TN,
                    K, lda, ldb, ldc);
            }
        }
    }
}

void test_correctness(
        const int M, const int N, const int K, const size_t mem_align) {

    assert(M > 0 && N > 0 && K > 0);

    size_t size_A = (size_t)M * (K + A_PAD) * sizeof(int8_t);
    size_t size_B = (size_t)K * (N + B_PAD) * sizeof(int8_t);
    size_t size_C = (size_t)M * (N + C_PAD) * sizeof(int32_t);

    int8_t *_A = (int8_t *)malloc(size_A + mem_align);
    int8_t *_B = (int8_t *)malloc(size_B + mem_align);
    int32_t *_C_amx = (int32_t *)malloc(size_C + mem_align);
    int32_t *_C_cpu = (int32_t *)malloc(size_C + mem_align);

    int8_t *A = (int8_t *)(((size_t)_A + mem_align) & ~(mem_align - 1));
    int8_t *B = (int8_t *)(((size_t)_B + mem_align) & ~(mem_align - 1));
    int32_t *C_amx = (int32_t *)(((size_t)_C_amx + mem_align) & ~(mem_align - 1));
    int32_t *C_cpu = (int32_t *)(((size_t)_C_cpu + mem_align) & ~(mem_align - 1));

    for (int i = 0; i < M * (K + A_PAD); i++) A[i] = rand() % 256;
    for (int i = 0; i < K * (N + B_PAD); i++) B[i] = rand() % 256;
    for (int i = 0; i < M * (N + C_PAD); i++) C_amx[i] = C_cpu[i] = 0xffffffff;

    // cpu_gemm_i8i8i32(A, B, C_amx, M, N, K, K + A_PAD, N + B_PAD, N + C_PAD);
    amx_gemm_i8i8i32_naive(A, B, C_cpu, M, N, K, K + A_PAD, N + B_PAD, N + C_PAD);
    amx_gemm_i8i8i32(A, B, C_amx, M, N, K, K + A_PAD, N + B_PAD, N + C_PAD);

    int correct = 0 == memcmp(C_cpu, C_amx, M * (N + C_PAD) * sizeof(int32_t));
    if (!correct) {
        for (uint32_t i = 0; i < M * (N + C_PAD); i++) {
            if (C_cpu[i] != C_amx[i]) {
                printf("Test Failed: M N K = %5d %5d %5d, "
                       "Mismatch at Index %d: %d != %d\n",
                       M, N, K, i, C_cpu[i], C_amx[i]);
                break;
            }
        }
    } else {
        printf("Test passed: M N K = %5d %5d %5d\n", M, N, K);
    }
    free(_A); free(_B); free(_C_amx); free(_C_cpu);
}

void test_performance(
        const int M, const int N, const int K,
        const size_t mem_align, const int num_repeats) {

    assert(M > 0 && N > 0 && K > 0);

    size_t size_A = (size_t)M * (K + A_PAD) * sizeof(int8_t);
    size_t size_B = (size_t)K * (N + B_PAD) * sizeof(int8_t);
    size_t size_C = (size_t)M * (N + C_PAD) * sizeof(int32_t);

    int8_t *_A = (int8_t *)malloc(size_A + mem_align);
    int8_t *_B = (int8_t *)malloc(size_B + mem_align);
    int32_t *_C = (int32_t *)malloc(size_C + mem_align);

    int8_t *A = (int8_t *)(((size_t)_A + mem_align) & ~(mem_align - 1));
    int8_t *B = (int8_t *)(((size_t)_B + mem_align) & ~(mem_align - 1));
    int32_t *C = (int32_t *)(((size_t)_C + mem_align) & ~(mem_align - 1));

    memset(A, 1, M * (K + A_PAD) * sizeof(int8_t));
    memset(B, 1, K * (N + B_PAD) * sizeof(int8_t));
    memset(C, 1, M * (N + C_PAD) * sizeof(int32_t));

    amx_gemm_i8i8i32(A, B, C, M, N, K, K + A_PAD, N + B_PAD, N + C_PAD); // warm up

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    for (uint32_t i = 0; i < num_repeats; i++)
        amx_gemm_i8i8i32(A, B, C, M, N, K, K + A_PAD, N + B_PAD, N + C_PAD);
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    uint64_t mac_count = (uint64_t)M * N * K * num_repeats;
    uint64_t ideal_mac_per_cycle = 1024 * CORES;
    double frequency = 2.3e9;

    uint64_t nanoseconds = (end_time.tv_sec - start_time.tv_sec) * 1e9 +
                           (end_time.tv_nsec - start_time.tv_nsec);
    double elapsed_time = (double)nanoseconds / 1e9;

    double utilization = ((double)mac_count / elapsed_time) /
            ideal_mac_per_cycle / frequency;
    double TOPS = (double)mac_count * 2 / 1e12 / elapsed_time;

    printf("M N K = %5d %5d %5d, Elapsed time = %4.6f s, "
           "Performance = %4.2f TOPS, Utilization = %3.2f%%\n",
            M, N, K, elapsed_time, TOPS, utilization * 100);

    free(_A); free(_B); free(_C);
}

int main() {
    __tilecfg tile_data = {0};
    // Request permission to linux kernel to run AMX
    if (!set_tiledata_use())
        exit(-1);
    init_tile_config(&tile_data);

    // paramters
    size_t mem_align = 4096;
    const int num_repeats = 10;

    // test correctness
    srand(time(0));
    for (int i = 0; i < 10; i++) {
        const int M_align = 32;
        const int N_align = 32;
        const int K_align = 64;
        int M = (rand() % 4096 + M_align) / M_align * M_align;
        int N = (rand() % 4096 + N_align) / N_align * N_align;
        int K = (rand() % 4096 + K_align) / K_align * K_align;
        test_correctness(M, N, K, mem_align);
    }

    // test performance
    printf("Test %d CPU Cores (%d Cores x %d Cores), ", CORES, CORES_M, CORES_N);
    if (IS_TEST_MNK) {
        printf("for M = N = K\n");
        for (int i = 256; i <= 16384; i += 256) {
            test_performance(i, i, i, mem_align, num_repeats);
            fflush(stdout);
        }
    }
    if (IS_TEST_K) {
        printf("for M N = %d %d\n", TEST_M, TEST_N);
        for (int i = 256; i <= 16384; i += 256) {
            test_performance(TEST_M, TEST_N, i, mem_align, num_repeats);
            fflush(stdout);
        }
    }

    return 0;
}