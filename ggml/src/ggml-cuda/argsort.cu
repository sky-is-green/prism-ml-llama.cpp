#include "argsort.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 1)
#        define STRIDED_ITERATOR_AVAILABLE
#        include <cuda/iterator>
#    endif
using namespace cub;
#endif  // GGML_CUDA_USE_CUB

static __global__ void init_indices(int * indices, const int ncols, const int nrows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (col < ncols && row < nrows) {
        indices[row * ncols + col] = col;
    }
}

#ifndef STRIDED_ITERATOR_AVAILABLE
static __global__ void init_offsets(int * offsets, const int ncols, const int nrows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= nrows) {
        offsets[idx] = idx * ncols;
    }
}
#endif  // STRIDED_ITERATOR_AVAILABLE

#ifdef GGML_CUDA_USE_CUB

// returns the suggested maximum number of rows to process during one argsort_f32_i32_cuda_cub() call
int argsort_f32_i32_cuda_cub_chunk_nrows(const size_t nb01, const int64_t nrows) {
    // perform argsort in chunks up to approximately this size (currently 64MB)
    // to avoid excessive temporary buffers memory usage
    const int chunk_bytes = 1 << 26;

    // calculate how many rows will fit in one chunk (must be at least one)
    const int chunk_nrows = std::max((int) (chunk_bytes / nb01), 1);

    // limit the resulting amount to total nrows
    return std::min((int64_t) chunk_nrows, nrows);
}

void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream) {
    ggml_cuda_pool_alloc<int>   temp_indices_alloc(pool, ncols * nrows);
    ggml_cuda_pool_alloc<float> temp_keys_alloc(pool, ncols * nrows);

    int *   temp_indices = temp_indices_alloc.get();
    float * temp_keys    = temp_keys_alloc.get();

    static const int block_size = 256;
    const dim3 grid_size((ncols + block_size - 1) / block_size, nrows);
    init_indices<<<grid_size, block_size, 0, stream>>>(temp_indices, ncols, nrows);

#ifdef STRIDED_ITERATOR_AVAILABLE
    auto offset_iterator = cuda::make_strided_iterator(cuda::make_counting_iterator(0), ncols);
#else
    // offset_iterator needs to populate nrows + 1 elements, so we also have to ceildiv nrows + 1 by block_size
    const int                 nrows_offset = nrows + 1;
    ggml_cuda_pool_alloc<int> offsets_alloc(pool, nrows_offset);
    int *                     offset_iterator = offsets_alloc.get();
    const dim3                offset_grid((nrows_offset + block_size - 1) / block_size);
    init_offsets<<<offset_grid, block_size, 0, stream>>>(offset_iterator, ncols, nrows);
#endif
    CUDA_CHECK(cudaMemcpyAsync(temp_keys, x, ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    size_t temp_storage_bytes = 0;

    bool is_capturing = false;
#ifdef USE_CUDA_GRAPH
    // Currently (confirmed for CCCL <= 3.2) DeviceSegmentedSort does not support stream capture, while DeviceSegmentedRadixSort does.
    // See https://github.com/NVIDIA/cccl/issues/5661#issuecomment-3229037149
    // TODO: constrain this to the CCCL versions that have this issue once it's resolved in a future CCCL release.
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    is_capturing = (capture_status != cudaStreamCaptureStatusNone);
#endif  // USE_CUDA_GRAPH

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, temp_keys, temp_keys,  // keys (in-place)
                                                  temp_indices, dst,  // values (indices)
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing || ncols > 1024) {   /* local: hipCUB's DeviceSegmentedSort fails with
                                                         "invalid configuration argument" for segment widths
                                                         past ~1024 (the DSA indexer's candidate count grows with
                                                         context).  The radix path handles any width and is stable. */
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(
                nullptr, temp_storage_bytes, temp_keys, temp_keys,  // keys (in-place)
                temp_indices, dst,                                  // values (indices)
                ncols * nrows, nrows,                               // num items, num segments
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairs(nullptr, temp_storage_bytes, temp_keys,
                                                      temp_keys,             // keys (in-place)
                                                      temp_indices, dst,     // values (indices)
                                                      ncols * nrows, nrows,  // num items, num segments
                                                      offset_iterator, offset_iterator + 1, stream));
        }
    } else {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys,
                                                            temp_keys,          // keys (in-place)
                                                            temp_indices, dst,  // values (indices)
                                                            ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing || ncols > 1024) {   /* local: hipCUB's DeviceSegmentedSort fails with
                                                         "invalid configuration argument" for segment widths
                                                         past ~1024 (the DSA indexer's candidate count grows with
                                                         context).  The radix path handles any width and is stable. */
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
                nullptr, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst, ncols * nrows, nrows,
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                                                temp_indices, dst, ncols * nrows, nrows,
                                                                offset_iterator, offset_iterator + 1, stream));
        }
    }

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys,
                                                  temp_keys,          // keys (in-place)
                                                  temp_indices, dst,  // values (indices)
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing || ncols > 1024) {   /* local: hipCUB's DeviceSegmentedSort fails with
                                                         "invalid configuration argument" for segment widths
                                                         past ~1024 (the DSA indexer's candidate count grows with
                                                         context).  The radix path handles any width and is stable. */
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                           temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                           offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                      temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                      offset_iterator + 1, stream));
        }
    } else {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys,
                                                            temp_keys,          // keys (in-place)
                                                            temp_indices, dst,  // values (indices)
                                                            ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing || ncols > 1024) {   /* local: hipCUB's DeviceSegmentedSort fails with
                                                         "invalid configuration argument" for segment widths
                                                         past ~1024 (the DSA indexer's candidate count grows with
                                                         context).  The radix path handles any width and is stable. */
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
                d_temp_storage, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst, ncols * nrows, nrows,
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys,
                                                                temp_keys, temp_indices, dst, ncols * nrows, nrows,
                                                                offset_iterator, offset_iterator + 1, stream));
        }
    }
}
#endif  // GGML_CUDA_USE_CUB

// Bitonic sort implementation
template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<ggml_sort_order order>
static __global__ void k_argsort_f32_i32(const float * x, int * dst, const int ncols, int ncols_pad) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

// Strata-DS4 (tools/v41/ggml/rank-argsort.patch): a row of <= 1024 columns sorted by RANK instead of the bitonic
// network below.  Each thread owns a column, counts the columns that sort before it (value order, equal values by
// ascending column) and writes itself at that rank: one pass over shared memory, one barrier.  The bitonic network
// takes 45 barrier steps for 384 -> 512 columns and reads x from global memory per compare: 9 us per 384-wide row on
// a 4090, and DeepSeek-V4.1 decode runs ~80 of them per token (router top-6 + prefetch top-16).  Equal values come out
// in ascending column order (the network leaves their order to its wiring).  Keys are the float bits mapped to an
// order-preserving uint (-0.0 before +0.0, NaNs at the ends), so the ranks always form a permutation.
// GGML_CUDA_RANK_ARGSORT=0 restores the bitonic kernel.
template<ggml_sort_order order>
static __global__ void k_argsort_rank_f32_i32(const float * __restrict__ x, int * __restrict__ dst, const int ncols,
                                              const int ncols4) {
    extern __shared__ uint4 s_key4[];   // ncols rounded up to 4
    unsigned * s_key = (unsigned *) s_key4;
    const int64_t row = blockIdx.x;
    const float * x_row = x + row * ncols;
    for (int c = threadIdx.x; c < ncols4; c += blockDim.x) {
        unsigned k = 0xFFFFFFFFu;   // padding: never before a real column (not smaller, and after it in column order)
        if (c < ncols) {
            const unsigned u = __float_as_uint(x_row[c]);
            k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // unsigned order = float order
            if (order == GGML_SORT_ORDER_DESC) {
                k = ~k;                                     // "sorts first" = "smaller key" in both orders
            }
        }
        s_key[c] = k;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < ncols; e += blockDim.x) {
        const unsigned ke = s_key[e];
        int r = 0;
        for (int f4 = 0; f4 < ncols4 / 4; ++f4) {
            const uint4 k4 = s_key4[f4];
            const int f = 4 * f4;
            r += (k4.x < ke) | ((k4.x == ke) & (f + 0 < e));
            r += (k4.y < ke) | ((k4.y == ke) & (f + 1 < e));
            r += (k4.z < ke) | ((k4.z == ke) & (f + 2 < e));
            r += (k4.w < ke) | ((k4.w == ke) & (f + 3 < e));
        }
        dst[row * ncols + r] = e;
    }
}

// Strata-DS4 (tools/v41/ggml/rank-argsort.patch): ggml_argsort_top_k records k in op_params[1] (only the first k
// columns of each sorted row are read through its view).  For k <= 32 of a row of <= 1024 one warp per row selects
// them directly: each lane holds its columns' keys (column lane + 32 i) as 64-bit (key << 32 | column) composites in
// registers, then k rounds of a 5-step shuffle minimum, the winning lane rescanning only its own columns.  The same
// first k as the rank kernel / a stable sort (value order, equal values by ascending column), ~1 us instead of a
// full sort of the row.  GGML_CUDA_RANK_ARGSORT=0 restores the bitonic kernel for every argsort.
template<ggml_sort_order order, int NPL>   // NPL: columns per lane, >= ceil(ncols / 32)
static __global__ void k_argsort_topk_warp_f32_i32(const float * __restrict__ x, int * __restrict__ dst, const int ncols,
                                                   const int k, const int nrows) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= nrows) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const float * x_row = x + (int64_t) row * ncols;
    unsigned long long key[NPL];
#pragma unroll
    for (int i = 0; i < NPL; ++i) {
        const int c = lane + 32 * i;
        unsigned long long kk = ~0ull;   // padding: after every real column
        if (c < ncols) {
            const unsigned u = __float_as_uint(x_row[c]);
            unsigned kv = (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // unsigned order = float order
            if (order == GGML_SORT_ORDER_DESC) {
                kv = ~kv;
            }
            kk = ((unsigned long long) kv << 32) | (unsigned) c;
        }
        key[i] = kk;
    }
    unsigned long long best = ~0ull;
#pragma unroll
    for (int i = 0; i < NPL; ++i) {
        best = key[i] < best ? key[i] : best;
    }
    int * dst_row = dst + (int64_t) row * ncols;
    for (int j = 0; j < k; ++j) {
        unsigned long long m = best;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            /* local: HIP's shim for __shfl_xor_sync takes a width (CUDA allows 3 args) */
            const unsigned long long o = __shfl_xor_sync(0xFFFFFFFFu, m, off, 32);
            m = o < m ? o : m;
        }
        const int col = (int) (m & 0xFFFFFFFFull);
        if (lane == 0) {
            dst_row[j] = col;
        }
        if (lane == (col & 31)) {   // the winner drops that column and rescans its own
            best = ~0ull;
#pragma unroll
            for (int i = 0; i < NPL; ++i) {
                if (i == (col >> 5)) {
                    key[i] = ~0ull;
                }
                best = key[i] < best ? key[i] : best;
            }
        }
    }
}

template<ggml_sort_order order>
static void argsort_topk_warp_launch(const float * x, int * dst, const int ncols, const int k, const int nrows,
                                     cudaStream_t stream) {
    const dim3 block_dims(128, 1, 1);   // 4 rows per block
    const dim3 block_nums((nrows + 3) / 4, 1, 1);
    const int npl = (ncols + 31) / 32;
#define TOPK_NPL(N) if (npl <= N) { k_argsort_topk_warp_f32_i32<order, N><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, k, nrows); return; }
    TOPK_NPL(1) TOPK_NPL(2) TOPK_NPL(4) TOPK_NPL(8) TOPK_NPL(12) TOPK_NPL(16) TOPK_NPL(24) TOPK_NPL(32)
#undef TOPK_NPL
    GGML_ABORT("fatal error");
}

static bool argsort_use_rank(const int ncols) {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_RANK_ARGSORT");
        return !(v && v[0] == '0');
    }();
    /* local: raise the ceiling so the DSA indexer's score rows (ncols grows with context) use the rank kernel
       instead of hipCUB's segmented sorts, which fail above ~1024 with "invalid configuration argument".
       4 bytes per column of shared memory: 16384 columns = 64 KiB, exactly gfx1100's limit. */
    return on && ncols <= 16384;
}

static void argsort_f32_i32_cuda_rank(const float * x, int * dst, const int ncols, const int nrows,
                                      ggml_sort_order order, cudaStream_t stream) {
    const int ncols4 = (ncols + 3) & ~3;
    const dim3 block_dims(std::min(1024, (ncols + 31) & ~31), 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols4 * sizeof(unsigned);
    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_rank_f32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols4);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_rank_f32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols4);
    } else {
        GGML_ABORT("fatal error");
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_ASC>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_DESC>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    if (argsort_use_rank((int) ncols)) {   // Strata-DS4: see k_argsort_rank_f32_i32 / k_argsort_topk_warp_f32_i32
        const int k_used = dst->op_params[1];   // ggml_argsort_top_k: only the first k of a row are read
        if (k_used > 0 && k_used <= 32 && k_used <= ncols) {
            if (order == GGML_SORT_ORDER_ASC) {
                argsort_topk_warp_launch<GGML_SORT_ORDER_ASC>(src0_d, (int *) dst_d, (int) ncols, k_used, (int) nrows, stream);
            } else {
                argsort_topk_warp_launch<GGML_SORT_ORDER_DESC>(src0_d, (int *) dst_d, (int) ncols, k_used, (int) nrows, stream);
            }
            return;
        }
        argsort_f32_i32_cuda_rank(src0_d, (int *) dst_d, (int) ncols, (int) nrows, order, stream);
        return;
    }

#ifdef GGML_CUDA_USE_CUB
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    // early return if we can use bitonic argsort
    if (shared_mem <= max_shared_mem && ncols <= 1024) {
        argsort_f32_i32_cuda_bitonic(src0_d, (int *) dst_d, ncols, nrows, order, stream);
        return;
    }

    const int chunk_nrows = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool & pool = ctx.pool();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        argsort_f32_i32_cuda_cub(pool, src0_d, (int *) dst_d, ncols, iter_nrows, order, stream);

        src0_d += ncols * iter_nrows;
        dst_d  += ncols * iter_nrows;
    }
#else
    argsort_f32_i32_cuda_bitonic(src0_d, (int *) dst_d, ncols, nrows, order, stream);
#endif
}
