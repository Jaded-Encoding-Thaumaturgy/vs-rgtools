__device__ static const int width = ${width};
__device__ static const int height = ${height};
__device__ static const float sigmaS = ${sigmaS};
__device__ static const float noise0 = ${noise0};
__device__ static const float noise1 = ${noise1};
__device__ static const int radius = ${radius};

__device__ static const int kernel_size_x = 2 * radius + ${block_x};
__device__ static const int kernel_size_y = 2 * radius + ${block_y};

extern "C"
__global__ void nabl(const ${data_type} * __restrict__ src, ${data_type} * __restrict__ dst) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    float num {};
    float den {};

    if (x >= width || y >= height)
        return;

    float center {};
    float value {};

    center = src[y * width + x];

    #pragma unroll 4
    for (int cy = max(y - radius, 0); cy <= min(y + radius, height - 1); ++cy) {
        #pragma unroll 4
        for (int cx = max(x - radius, 0); cx <= min(x + radius, width - 1); ++cx) {
            value = src[cy * width + cx];

            float noise_g = noise0 * value + noise1;
            float euc_dist = sqrtf((cy - y) * (cy - y) + (cx - x) * (cx - x));

            float weight = expf((euc_dist * euc_dist) / (-2 * (sigmaS * sigmaS))) * (
                expf(-2 * noise_g) * jnf(sqrtf((value - center) * value - center), 2 * noise_g)
            );

            num += weight * value;
            den += weight;
        }
    }

    dst[y * width + x] = num / den;
}