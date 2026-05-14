// TurboQuant KV-cache compression kernels.
//
// GPU mirror of the CPU encoder/decoder in ../ds4_turbo.c. Operates on
// head_dim = 512 rows, four 128-float Walsh-Hadamard rotations per row,
// 16 quantization blocks of 32 floats per row. The sign masks (two
// constant +/-1 vectors derived from seed 42 / 43 in the CPU code) are
// inlined below so the GPU result matches the CPU bit-for-bit.
//
// Encoded layout per 32-float block:
//   - 3-bit: 2B norm + 8B qs (low 2 bits) + 4B signs (bit 2)            = 14 B
//   - 4-bit: 2B norm + 8B qs + 4B signs + 4B qjl (bit 3)                = 18 B
//   - 6-bit: 2B norm + 24B qs (4-index groups packed 24 bits / 3 bytes) = 26 B
//   - 8-bit: 2B norm + 32B qs (one byte per signed codebook index)      = 34 B
//
// The 3-bit format is a strict prefix of the 4-bit format; the qjl
// slot only carries the high bit of the 4-bit codebook index. The 6/8-bit
// formats use signed codebooks (Lloyd-Max for N(0,1) spans both signs),
// so no separate sign bits are required.

#define TURBO_ROT_D    128
#define TURBO_BLOCK    32

struct ds4_metal_args_turbo {
    int32_t  head_dim;
    int32_t  bits;          // 3/4/6/8 (for wht_inplace: 0 = forward, 1 = inverse)
    int32_t  n_rows;        // batch dispatcher; per-row kernels ignore this
    int32_t  pad;
    uint64_t row_bytes_in;  // encoded row stride in bytes (batch decode)
    uint64_t row_bytes_out; // output row stride in bytes (batch decode)
};

// Lloyd-Max optimal centroids for N(0, 1).
constant float turbo_cb3[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f,
};

constant float turbo_cb3_mid[7] = {
    -1.7479f, -1.0499f, -0.5005f, 0.0000f,
     0.5005f,  1.0499f,  1.7479f,
};

constant float turbo_cb4[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f,
};

constant float turbo_cb4_mid[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f,
    -0.7995f, -0.5224f, -0.2582f,  0.0000f,
     0.2582f,  0.5224f,  0.7995f,  1.0993f,
     1.4371f,  1.8435f,  2.4008f,
};

// Lloyd-Max optimal centroids for N(0, 1), 64 levels (6-bit codebook).
constant float turbo_cb6[64] = {
    -3.605999f, -3.085722f, -2.751095f, -2.496953f, -2.288778f, -2.110705f, -1.954065f, -1.813578f,
    -1.685776f, -1.568258f, -1.459279f, -1.357532f, -1.262008f, -1.171905f, -1.086573f, -1.005472f,
    -0.928148f, -0.854207f, -0.783309f, -0.715146f, -0.649445f, -0.585951f, -0.524433f, -0.464669f,
    -0.406453f, -0.349584f, -0.293873f, -0.239131f, -0.185179f, -0.131837f, -0.078929f, -0.026281f,
     0.026281f,  0.078929f,  0.131837f,  0.185179f,  0.239131f,  0.293873f,  0.349584f,  0.406453f,
     0.464669f,  0.524433f,  0.585951f,  0.649445f,  0.715146f,  0.783309f,  0.854207f,  0.928148f,
     1.005472f,  1.086573f,  1.171905f,  1.262008f,  1.357532f,  1.459279f,  1.568258f,  1.685776f,
     1.813578f,  1.954065f,  2.110705f,  2.288778f,  2.496953f,  2.751095f,  3.085722f,  3.605999f,
};

constant float turbo_cb6_mid[63] = {
    -3.345860f, -2.918408f, -2.624024f, -2.392865f, -2.199742f, -2.032385f, -1.883821f, -1.749677f,
    -1.627017f, -1.513768f, -1.408405f, -1.309770f, -1.216957f, -1.129239f, -1.046022f, -0.966810f,
    -0.891178f, -0.818758f, -0.749228f, -0.682295f, -0.617698f, -0.555192f, -0.494551f, -0.435561f,
    -0.378018f, -0.321728f, -0.266502f, -0.212155f, -0.158508f, -0.105383f, -0.052605f,  0.000000f,
     0.052605f,  0.105383f,  0.158508f,  0.212155f,  0.266502f,  0.321728f,  0.378018f,  0.435561f,
     0.494551f,  0.555192f,  0.617698f,  0.682295f,  0.749228f,  0.818758f,  0.891178f,  0.966810f,
     1.046022f,  1.129239f,  1.216957f,  1.309770f,  1.408405f,  1.513768f,  1.627017f,  1.749677f,
     1.883821f,  2.032385f,  2.199742f,  2.392865f,  2.624024f,  2.918408f,  3.345860f,
};

// Lloyd-Max optimal centroids for N(0, 1), 256 levels (8-bit codebook).
constant float turbo_cb8[256] = {
    -4.035480f, -3.565625f, -3.268187f, -3.045475f, -2.865491f, -2.713551f, -2.581644f, -2.464895f,
    -2.360107f, -2.265066f, -2.178166f, -2.098206f, -2.024257f, -1.955584f, -1.891595f, -1.831799f,
    -1.775785f, -1.723203f, -1.673751f, -1.627164f, -1.583207f, -1.541672f, -1.502368f, -1.465126f,
    -1.429789f, -1.396212f, -1.364264f, -1.333822f, -1.304772f, -1.277010f, -1.250438f, -1.224965f,
    -1.200508f, -1.176989f, -1.154335f, -1.132480f, -1.111361f, -1.090923f, -1.071113f, -1.051883f,
    -1.033188f, -1.014988f, -0.997247f, -0.979930f, -0.963006f, -0.946448f, -0.930229f, -0.914327f,
    -0.898719f, -0.883388f, -0.868315f, -0.853484f, -0.838881f, -0.824492f, -0.810305f, -0.796310f,
    -0.782495f, -0.768852f, -0.755371f, -0.742046f, -0.728869f, -0.715832f, -0.702931f, -0.690157f,
    -0.677508f, -0.664976f, -0.652557f, -0.640248f, -0.628042f, -0.615938f, -0.603930f, -0.592014f,
    -0.580189f, -0.568449f, -0.556793f, -0.545217f, -0.533718f, -0.522294f, -0.510941f, -0.499658f,
    -0.488442f, -0.477290f, -0.466201f, -0.455172f, -0.444200f, -0.433285f, -0.422424f, -0.411614f,
    -0.400855f, -0.390145f, -0.379481f, -0.368862f, -0.358286f, -0.347752f, -0.337259f, -0.326803f,
    -0.316386f, -0.306003f, -0.295655f, -0.285340f, -0.275057f, -0.264803f, -0.254579f, -0.244382f,
    -0.234211f, -0.224066f, -0.213944f, -0.203846f, -0.193768f, -0.183712f, -0.173674f, -0.163654f,
    -0.153652f, -0.143665f, -0.133694f, -0.123736f, -0.113791f, -0.103857f, -0.093934f, -0.084021f,
    -0.074116f, -0.064219f, -0.054328f, -0.044443f, -0.034562f, -0.024685f, -0.014810f, -0.004936f,
     0.004936f,  0.014810f,  0.024685f,  0.034562f,  0.044443f,  0.054328f,  0.064219f,  0.074116f,
     0.084021f,  0.093934f,  0.103857f,  0.113791f,  0.123736f,  0.133694f,  0.143665f,  0.153652f,
     0.163654f,  0.173674f,  0.183712f,  0.193768f,  0.203846f,  0.213944f,  0.224066f,  0.234211f,
     0.244382f,  0.254579f,  0.264803f,  0.275057f,  0.285340f,  0.295655f,  0.306003f,  0.316386f,
     0.326803f,  0.337259f,  0.347752f,  0.358286f,  0.368862f,  0.379481f,  0.390145f,  0.400855f,
     0.411614f,  0.422424f,  0.433285f,  0.444200f,  0.455172f,  0.466201f,  0.477290f,  0.488442f,
     0.499658f,  0.510941f,  0.522294f,  0.533718f,  0.545217f,  0.556793f,  0.568449f,  0.580189f,
     0.592014f,  0.603930f,  0.615938f,  0.628042f,  0.640248f,  0.652557f,  0.664976f,  0.677508f,
     0.690157f,  0.702931f,  0.715832f,  0.728869f,  0.742046f,  0.755371f,  0.768852f,  0.782495f,
     0.796310f,  0.810305f,  0.824492f,  0.838881f,  0.853484f,  0.868315f,  0.883388f,  0.898719f,
     0.914327f,  0.930229f,  0.946448f,  0.963006f,  0.979930f,  0.997247f,  1.014988f,  1.033188f,
     1.051883f,  1.071113f,  1.090923f,  1.111361f,  1.132480f,  1.154335f,  1.176989f,  1.200508f,
     1.224965f,  1.250438f,  1.277010f,  1.304772f,  1.333822f,  1.364264f,  1.396212f,  1.429789f,
     1.465126f,  1.502368f,  1.541672f,  1.583207f,  1.627164f,  1.673751f,  1.723203f,  1.775785f,
     1.831799f,  1.891595f,  1.955584f,  2.024257f,  2.098206f,  2.178166f,  2.265066f,  2.360107f,
     2.464895f,  2.581644f,  2.713551f,  2.865491f,  3.045475f,  3.268187f,  3.565625f,  4.035480f,
};

constant float turbo_cb8_mid[255] = {
    -3.800552f, -3.416906f, -3.156831f, -2.955483f, -2.789521f, -2.647598f, -2.523269f, -2.412501f,
    -2.312586f, -2.221616f, -2.138186f, -2.061231f, -1.989920f, -1.923590f, -1.861697f, -1.803792f,
    -1.749494f, -1.698477f, -1.650457f, -1.605185f, -1.562439f, -1.522020f, -1.483747f, -1.447458f,
    -1.413001f, -1.380238f, -1.349043f, -1.319297f, -1.290891f, -1.263724f, -1.237702f, -1.212737f,
    -1.188749f, -1.165662f, -1.143407f, -1.121920f, -1.101142f, -1.081018f, -1.061498f, -1.042535f,
    -1.024088f, -1.006118f, -0.988588f, -0.971468f, -0.954727f, -0.938338f, -0.922278f, -0.906523f,
    -0.891054f, -0.875851f, -0.860899f, -0.846182f, -0.831686f, -0.817398f, -0.803307f, -0.789402f,
    -0.775673f, -0.762112f, -0.748709f, -0.735458f, -0.722351f, -0.709382f, -0.696544f, -0.683833f,
    -0.671242f, -0.658767f, -0.646402f, -0.634145f, -0.621990f, -0.609934f, -0.597972f, -0.586102f,
    -0.574319f, -0.562621f, -0.551005f, -0.539468f, -0.528006f, -0.516618f, -0.505300f, -0.494050f,
    -0.482866f, -0.471746f, -0.460686f, -0.449686f, -0.438743f, -0.427854f, -0.417019f, -0.406235f,
    -0.395500f, -0.384813f, -0.374171f, -0.363574f, -0.353019f, -0.342505f, -0.332031f, -0.321594f,
    -0.311194f, -0.300829f, -0.290498f, -0.280198f, -0.269930f, -0.259691f, -0.249480f, -0.239297f,
    -0.229139f, -0.219005f, -0.208895f, -0.198807f, -0.188740f, -0.178693f, -0.168664f, -0.158653f,
    -0.148659f, -0.138680f, -0.128715f, -0.118763f, -0.108824f, -0.098896f, -0.088977f, -0.079068f,
    -0.069167f, -0.059273f, -0.049385f, -0.039502f, -0.029623f, -0.019747f, -0.009873f,  0.000000f,
     0.009873f,  0.019747f,  0.029623f,  0.039502f,  0.049385f,  0.059273f,  0.069167f,  0.079068f,
     0.088977f,  0.098896f,  0.108824f,  0.118763f,  0.128715f,  0.138680f,  0.148659f,  0.158653f,
     0.168664f,  0.178693f,  0.188740f,  0.198807f,  0.208895f,  0.219005f,  0.229139f,  0.239297f,
     0.249480f,  0.259691f,  0.269930f,  0.280198f,  0.290498f,  0.300829f,  0.311194f,  0.321594f,
     0.332031f,  0.342505f,  0.353019f,  0.363574f,  0.374171f,  0.384813f,  0.395500f,  0.406235f,
     0.417019f,  0.427854f,  0.438743f,  0.449686f,  0.460686f,  0.471746f,  0.482866f,  0.494050f,
     0.505300f,  0.516618f,  0.528006f,  0.539468f,  0.551005f,  0.562621f,  0.574319f,  0.586102f,
     0.597972f,  0.609934f,  0.621990f,  0.634145f,  0.646402f,  0.658767f,  0.671242f,  0.683833f,
     0.696544f,  0.709382f,  0.722351f,  0.735458f,  0.748709f,  0.762112f,  0.775673f,  0.789402f,
     0.803307f,  0.817398f,  0.831686f,  0.846182f,  0.860899f,  0.875851f,  0.891054f,  0.906523f,
     0.922278f,  0.938338f,  0.954727f,  0.971468f,  0.988588f,  1.006118f,  1.024088f,  1.042535f,
     1.061498f,  1.081018f,  1.101142f,  1.121920f,  1.143407f,  1.165662f,  1.188749f,  1.212737f,
     1.237702f,  1.263724f,  1.290891f,  1.319297f,  1.349043f,  1.380238f,  1.413001f,  1.447458f,
     1.483747f,  1.522020f,  1.562439f,  1.605185f,  1.650457f,  1.698477f,  1.749494f,  1.803792f,
     1.861697f,  1.923590f,  1.989920f,  2.061231f,  2.138186f,  2.221616f,  2.312586f,  2.412501f,
     2.523269f,  2.647598f,  2.789521f,  2.955483f,  3.156831f,  3.416906f,  3.800552f,
};

// WHT sign masks. These are the outputs of turbo_fill_signs() with
// seeds 42 and 43, generated once and committed verbatim so the GPU
// never has to rerun the LCG. Any change to the CPU seed or LCG must
// update both sides simultaneously.
constant float turbo_wht_signs1[TURBO_ROT_D] = {
    -1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f,
    -1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f,
    -1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f,
    -1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,
     1.f,  1.f,  1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f,
    -1.f, -1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f,
    -1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f,
     1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f,
};

constant float turbo_wht_signs2[TURBO_ROT_D] = {
     1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f,  1.f,
     1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f,
    -1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,  1.f,
     1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f,
     1.f, -1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f,
     1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f,  1.f,
     1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,
     1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f, -1.f,
};

// In-place fast Walsh-Hadamard transform on 128 floats living in
// threadgroup memory. Each thread owns one lane (tid in [0, 128)).
//
// At stage h, the butterfly pairs lanes (j, j+h) where j has bit `h`
// clear. With the XOR trick `partner = tid ^ h`, lower lanes (tid <
// partner) want a+b and upper lanes want b-a, where a is the original
// value at tid and b is the original value at partner. We read both
// values, barrier, then write so reads happen before any write.
//
// The CPU applies a 1/sqrt(128) scale at the end; we fuse that as a
// final per-lane multiply after the seven butterfly stages.
static void turbo_fwht_128_tg(threadgroup float *x, ushort tid) {
    for (ushort h = 1; h < TURBO_ROT_D; h <<= 1) {
        const ushort pair = tid ^ h;
        const float a = x[tid];
        const float b = x[pair];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        x[tid] = (tid < pair) ? (a + b) : (b - a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 0.08838834764831845f; // 1 / sqrt(128)
    x[tid] *= inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Nearest-centroid by midpoint scan. Codebook is sorted; the linear
// scan stays branch-friendly without the recurrence of the CPU loop.
static int turbo_nearest3(float v) {
    int i = 0;
    for (int k = 0; k < 7; k++) {
        if (v >= turbo_cb3_mid[k]) i = k + 1;
    }
    return i;
}

static int turbo_nearest4(float v) {
    int i = 0;
    for (int k = 0; k < 15; k++) {
        if (v >= turbo_cb4_mid[k]) i = k + 1;
    }
    return i;
}

// 6/8-bit nearest-centroid. Signed codebooks, single linear scan over
// all thresholds. Keeps the GPU mirror branch-light at the cost of a
// 63 / 255 iteration loop per lane; both stay short next to the WHT
// butterfly stages.
static int turbo_nearest6(float v) {
    int i = 0;
    for (int k = 0; k < 63; k++) {
        if (v >= turbo_cb6_mid[k]) i = k + 1;
    }
    return i;
}

static int turbo_nearest8(float v) {
    int i = 0;
    for (int k = 0; k < 255; k++) {
        if (v >= turbo_cb8_mid[k]) i = k + 1;
    }
    return i;
}

// IEEE 754 binary16 conversion via Metal's native half type.
static ushort turbo_f32_to_f16_bits(float f) {
    return as_type<ushort>((half)f);
}

static float turbo_f16_to_f32_bits(ushort h) {
    return (float) as_type<half>(h);
}

// Per-block packing. Each 32-lane simdgroup quantizes one block, then
// four lanes per block (0..3) cooperatively assemble the packed bytes.
// We avoid simd_or() so this compiles on older Metal toolchains;
// instead we serialize the OR through scratch with a 32-element
// reduction.
//
// scratch layout per kernel call (per row):
//   [0..3]    : per-block norm sq reduction final slot (reused)
//   [4..131]  : per-lane squared-value scratch for L2 reduction (128 floats)
//   [136..648]: per-lane packed-bits staging (128 lanes * 4 uints = 512 floats)
//   [656..671]: per-block packed words (4 blocks * 4 uints = 16 floats)
//
// All offsets are in float units; uints are aliased via cast.
kernel void kernel_ds4_turbo_quantize_row(
        constant ds4_metal_args_turbo & args,
        device const float            * src,
        device       uchar            * dst,
        threadgroup  float            * rot     [[threadgroup(0)]],
        threadgroup  float            * scratch [[threadgroup(1)]],
        uint  row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]]) {
    const int head_dim = args.head_dim;
    const int bits     = args.bits;
    if (head_dim <= 0 || (head_dim % TURBO_ROT_D) != 0) return;
    if (bits != 3 && bits != 4 && bits != 6 && bits != 8) return;

    const int n_rot     = head_dim / TURBO_ROT_D;
    const int blocks_pg = TURBO_ROT_D / TURBO_BLOCK; // 4
    uint blk_bytes;
    if      (bits == 3) blk_bytes = 14u;
    else if (bits == 4) blk_bytes = 18u;
    else if (bits == 6) blk_bytes = 26u;
    else                blk_bytes = 34u;

    device const float * src_row = src + (uint)row * (uint)head_dim;
    device       uchar * dst_row = dst + (uint)row * (uint)blk_bytes * (uint)(n_rot * blocks_pg);

    const ushort b    = tid >> 5;    // 0..3 (block index)
    const ushort lane = tid & 31;    // 0..31 (lane within block)

    threadgroup float * sq_buf      = scratch + 4;
    threadgroup uint  * stage       = (threadgroup uint *)(scratch + 136);
    threadgroup uint  * pack        = (threadgroup uint *)(scratch + 656);
    // For 6/8-bit, reuse the per-lane stage area to hold the raw index
    // (one int per lane). 128 lanes * 4 bytes fits well below the 512
    // slots `stage` already occupies.
    threadgroup int   * idx_buf = (threadgroup int *)stage;

    for (int g = 0; g < n_rot; g++) {
        // Load one rotation group; pre-multiply by signs1.
        rot[tid] = src_row[g * TURBO_ROT_D + tid] * turbo_wht_signs1[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        turbo_fwht_128_tg(rot, tid);

        // Post-multiply by signs2.
        rot[tid] *= turbo_wht_signs2[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-block L2 norm. Each lane squares its value into sq_buf
        // at its own index; then each 32-lane block runs a tree
        // reduction in place.
        sq_buf[tid] = rot[tid] * rot[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort s = 16; s > 0; s >>= 1) {
            if (lane < s) sq_buf[tid] += sq_buf[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float norm_sq = sq_buf[b * 32];
        const float norm    = sqrt(norm_sq);
        const float scale   = (norm > 1e-12f) ? (sqrt((float)TURBO_BLOCK) / norm) : 0.0f;

        // Quantize.
        const float val = rot[tid] * scale;
        int idx;
        if      (bits == 3) idx = turbo_nearest3(val);
        else if (bits == 4) idx = turbo_nearest4(val);
        else if (bits == 6) idx = turbo_nearest6(val);
        else                idx = turbo_nearest8(val);

        if (bits == 3 || bits == 4) {
            // 3/4-bit: bit-sliced per-lane contributions ORed into 4
            // packed uints per block (qs_lo, qs_hi, signs, qjl).
            const uint v_lo = (uint)(idx & 0x3);
            const uint v_b2 = (uint)((idx >> 2) & 0x1);
            const uint v_b3 = (uint)((idx >> 3) & 0x1);

            const uint qs_word_lo = (lane <  16) ? (v_lo << ((lane & 15) * 2)) : 0u;
            const uint qs_word_hi = (lane >= 16) ? (v_lo << ((lane & 15) * 2)) : 0u;
            const uint sg_word    = v_b2 << lane;
            const uint qj_word    = v_b3 << lane;

            stage[tid * 4 + 0] = qs_word_lo;
            stage[tid * 4 + 1] = qs_word_hi;
            stage[tid * 4 + 2] = sg_word;
            stage[tid * 4 + 3] = qj_word;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // OR-reduce within each 32-lane block: lanes 0..3 of each
            // block fold the 32 contributions for their packed slot.
            if (lane < 4) {
                uint acc = 0u;
                for (int j = 0; j < 32; j++) {
                    acc |= stage[(b * 32 + j) * 4 + lane];
                }
                pack[b * 4 + lane] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // One writer per block emits the encoded bytes.
            if (lane == 0) {
                const int block_idx = g * blocks_pg + b;
                device uchar * blk_out = dst_row + (uint)block_idx * blk_bytes;

                const ushort n16 = turbo_f32_to_f16_bits(norm);
                blk_out[0] = (uchar)(n16 & 0xff);
                blk_out[1] = (uchar)((n16 >> 8) & 0xff);

                const uint qs_lo = pack[b * 4 + 0];
                const uint qs_hi = pack[b * 4 + 1];
                const uint sg    = pack[b * 4 + 2];
                const uint qj    = pack[b * 4 + 3];

                blk_out[2] = (uchar)( qs_lo        & 0xff);
                blk_out[3] = (uchar)((qs_lo >>  8) & 0xff);
                blk_out[4] = (uchar)((qs_lo >> 16) & 0xff);
                blk_out[5] = (uchar)((qs_lo >> 24) & 0xff);
                blk_out[6] = (uchar)( qs_hi        & 0xff);
                blk_out[7] = (uchar)((qs_hi >>  8) & 0xff);
                blk_out[8] = (uchar)((qs_hi >> 16) & 0xff);
                blk_out[9] = (uchar)((qs_hi >> 24) & 0xff);
                blk_out[10] = (uchar)( sg        & 0xff);
                blk_out[11] = (uchar)((sg >>  8) & 0xff);
                blk_out[12] = (uchar)((sg >> 16) & 0xff);
                blk_out[13] = (uchar)((sg >> 24) & 0xff);
                if (bits == 4) {
                    blk_out[14] = (uchar)( qj        & 0xff);
                    blk_out[15] = (uchar)((qj >>  8) & 0xff);
                    blk_out[16] = (uchar)((qj >> 16) & 0xff);
                    blk_out[17] = (uchar)((qj >> 24) & 0xff);
                }
            }
        } else {
            // 6/8-bit: each lane stashes its full index, then a small
            // crew of lanes per block writes the packed payload bytes.
            // No bit-slicing -- 6-bit packs 4 indices per 3 bytes, 8-bit
            // is one byte per index.
            idx_buf[tid] = idx;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (lane == 0) {
                const int block_idx = g * blocks_pg + b;
                device uchar * blk_out = dst_row + (uint)block_idx * blk_bytes;

                const ushort n16 = turbo_f32_to_f16_bits(norm);
                blk_out[0] = (uchar)(n16 & 0xff);
                blk_out[1] = (uchar)((n16 >> 8) & 0xff);

                if (bits == 8) {
                    // Trivial: 32 bytes, one per index.
                    for (int i = 0; i < TURBO_BLOCK; i++) {
                        blk_out[2 + i] = (uchar)(idx_buf[b * 32 + i] & 0xff);
                    }
                } else {
                    // 6-bit: pack groups of 4 indices into 3 bytes.
                    // word = a | (b << 6) | (c << 12) | (d << 18).
                    for (int gi = 0; gi < TURBO_BLOCK / 4; gi++) {
                        const uint a = (uint)(idx_buf[b * 32 + 4*gi + 0] & 0x3f);
                        const uint bb = (uint)(idx_buf[b * 32 + 4*gi + 1] & 0x3f);
                        const uint c = (uint)(idx_buf[b * 32 + 4*gi + 2] & 0x3f);
                        const uint d = (uint)(idx_buf[b * 32 + 4*gi + 3] & 0x3f);
                        const uint w = a | (bb << 6) | (c << 12) | (d << 18);
                        blk_out[2 + 3*gi + 0] = (uchar)( w        & 0xff);
                        blk_out[2 + 3*gi + 1] = (uchar)((w >>  8) & 0xff);
                        blk_out[2 + 3*gi + 2] = (uchar)((w >> 16) & 0xff);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Dequantize one rotation group from packed bytes into the threadgroup
// `rot` buffer, then apply the inverse rotation in place. Each thread
// owns lane `tid`. The block layout matches turbo_unpack_blockN in
// ds4_turbo.c verbatim.
//
// `norms_tg` is a 4-slot threadgroup buffer used to broadcast each
// block's decoded fp16 norm to the 32 lanes that need it.
static void turbo_dequant_group(
        device const uchar      * row_in,
        threadgroup float       * rot,
        threadgroup float       * norms_tg,
        int g,
        int bits,
        ushort tid) {
    const int blocks_pg = TURBO_ROT_D / TURBO_BLOCK;
    uint blk_bytes;
    if      (bits == 3) blk_bytes = 14u;
    else if (bits == 4) blk_bytes = 18u;
    else if (bits == 6) blk_bytes = 26u;
    else                blk_bytes = 34u;

    const ushort b    = tid >> 5;
    const ushort lane = tid & 31;

    const int block_idx = g * blocks_pg + b;
    device const uchar * blk = row_in + (uint)block_idx * blk_bytes;

    if (lane == 0) {
        const ushort n16 = (ushort)blk[0] | ((ushort)blk[1] << 8);
        norms_tg[b] = turbo_f16_to_f32_bits(n16);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_scale = norms_tg[b] * (1.0f / sqrt((float)TURBO_BLOCK));

    int idx;
    float c;
    if (bits == 3 || bits == 4) {
        const uchar qs_byte = blk[2 + (lane >> 2)];
        const uchar sg_byte = blk[10 + (lane >> 3)];
        const uchar lo      = (qs_byte >> ((lane & 3) * 2)) & 0x3;
        const uchar hi      = (sg_byte >> (lane & 7)) & 0x1;
        if (bits == 3) {
            idx = (int)(lo | (hi << 2));
            c = turbo_cb3[idx];
        } else {
            const uchar qj_byte = blk[14 + (lane >> 3)];
            const uchar hi2 = (qj_byte >> (lane & 7)) & 0x1;
            idx = (int)(lo | (hi << 2) | (hi2 << 3));
            c = turbo_cb4[idx];
        }
    } else if (bits == 6) {
        // Each 4-lane group decodes 3 bytes back into 4 6-bit indices.
        const ushort gi = lane >> 2;    // group within block (0..7)
        const ushort sl = lane & 3;     // lane within group (0..3)
        const uint base = 2u + (uint)gi * 3u;
        const uint w =
            (uint)blk[base + 0]        |
            ((uint)blk[base + 1] <<  8) |
            ((uint)blk[base + 2] << 16);
        idx = (int)((w >> (sl * 6)) & 0x3fu);
        c = turbo_cb6[idx];
    } else {
        // 8-bit: one byte per index.
        idx = (int)blk[2 + lane];
        c = turbo_cb8[idx];
    }

    rot[tid] = c * inv_scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inverse rotation: signs2 -> FWHT -> signs1. WHT is self-inverse
    // modulo the 1/sqrt(128) scale that fwht_128_tg already applies.
    rot[tid] *= turbo_wht_signs2[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    turbo_fwht_128_tg(rot, tid);

    rot[tid] *= turbo_wht_signs1[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Decode one row to fp16.
kernel void kernel_ds4_turbo_dequantize_row(
        constant ds4_metal_args_turbo & args,
        device const uchar            * src,
        device       half             * dst,
        threadgroup  float            * rot     [[threadgroup(0)]],
        threadgroup  float            * scratch [[threadgroup(1)]],
        uint  row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]]) {
    const int head_dim = args.head_dim;
    const int bits     = args.bits;
    if (head_dim <= 0 || (head_dim % TURBO_ROT_D) != 0) return;
    if (bits != 3 && bits != 4 && bits != 6 && bits != 8) return;

    const int n_rot     = head_dim / TURBO_ROT_D;
    const int blocks_pg = TURBO_ROT_D / TURBO_BLOCK;
    uint blk_bytes;
    if      (bits == 3) blk_bytes = 14u;
    else if (bits == 4) blk_bytes = 18u;
    else if (bits == 6) blk_bytes = 26u;
    else                blk_bytes = 34u;
    const uint row_bytes = (uint)blk_bytes * (uint)(n_rot * blocks_pg);

    device const uchar * row_in  = src + (uint)row * row_bytes;
    device       half  * row_out = dst + (uint)row * (uint)head_dim;

    for (int g = 0; g < n_rot; g++) {
        turbo_dequant_group(row_in, rot, scratch, g, bits, tid);
        row_out[g * TURBO_ROT_D + tid] = (half)rot[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Batch decode. Grid is (n_rows, 1, 1), threadgroup is (128, 1, 1).
// Row strides are taken from args so the caller can pack rows tightly
// (row_bytes_in = head_dim/32 * blk_bytes) or with padding.
kernel void kernel_ds4_turbo_dequantize_batch(
        constant ds4_metal_args_turbo & args,
        device const uchar            * src,
        device       half             * dst,
        threadgroup  float            * rot     [[threadgroup(0)]],
        threadgroup  float            * scratch [[threadgroup(1)]],
        uint  row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]]) {
    if ((int)row >= args.n_rows) return;

    const int head_dim = args.head_dim;
    const int bits     = args.bits;
    if (head_dim <= 0 || (head_dim % TURBO_ROT_D) != 0) return;
    if (bits != 3 && bits != 4 && bits != 6 && bits != 8) return;

    const int n_rot          = head_dim / TURBO_ROT_D;
    const uint row_bytes_in  = (uint)args.row_bytes_in;
    const uint dst_stride    = (uint)(args.row_bytes_out / sizeof(half));

    device const uchar * row_in  = src + (uint)row * row_bytes_in;
    device       half  * row_out = dst + (uint)row * dst_stride;

    for (int g = 0; g < n_rot; g++) {
        turbo_dequant_group(row_in, rot, scratch, g, bits, tid);
        row_out[g * TURBO_ROT_D + tid] = (half)rot[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// KV-cache RoPE tail glue.
//
// The CPU TurboQuant KV layout stores the last n_rot dims of each
// 512-float row as raw fp16 to preserve the rotary positional
// encoding bit-for-bit: turbo cannot be trusted to round-trip the
// trigonometric tail with enough fidelity for attention to locate
// tokens. The GPU mirrors this layout: each cache row is
//   [ turbo_bytes(512) | n_rot * fp16 RoPE tail ]
// where the turbo block still encodes all 512 dims (so the rotation
// groups are aligned to 128-dim boundaries), and the fp16 tail
// overlays the last n_rot dims on read.
//
// Encode tail: lanes 0..n_rot-1 read the last n_rot fp32 source
// values and store them as fp16 at the row's tail slot.
//
// args.head_dim   = 512
// args.bits       = 3 or 4 (selects the turbo prefix size)
// args.n_rot      = number of RoPE dims (passed via args.pad)
// args.row_bytes_in  = full row stride in dst (turbo + tail)
// args.row_bytes_out = source fp32 row stride in bytes
//
// Source layout: tightly packed `n_rows * head_dim` floats.
// Dst layout:    `n_rows * row_bytes_in` bytes; each row's tail
//                lives at byte offset (row_bytes_in - n_rot*2).
kernel void kernel_ds4_turbo_kv_rope_tail_encode(
        constant ds4_metal_args_turbo & args,
        device const float            * src,
        device       uchar            * dst,
        uint  row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]]) {
    if ((int)row >= args.n_rows) return;
    const int head_dim = args.head_dim;
    const int n_rot    = args.pad;       // reuse the pad slot for n_rot
    if (head_dim <= 0 || n_rot <= 0 || n_rot > head_dim) return;
    if ((int)tid >= n_rot) return;

    const uint src_stride = (uint)(args.row_bytes_out / sizeof(float));
    const uint dst_stride = (uint)args.row_bytes_in;
    const uint tail_off   = dst_stride - (uint)n_rot * (uint)sizeof(half);

    const uint nope = (uint)(head_dim - n_rot);
    const float v = src[(uint)row * src_stride + nope + (uint)tid];
    const ushort h = turbo_f32_to_f16_bits(v);

    device uchar * tail = dst + (uint)row * dst_stride + tail_off;
    tail[(uint)tid * 2 + 0] = (uchar)(h & 0xff);
    tail[(uint)tid * 2 + 1] = (uchar)((h >> 8) & 0xff);
}

// Decode tail: overwrite the last n_rot halves of each decoded row
// in `dst` with the fp16 tail bytes from `src`. Runs after the
// turbo dequant kernel has populated the full 512-half row; the
// turbo-decoded values in the RoPE region are discarded.
//
// args.head_dim   = 512
// args.n_rot      = number of RoPE dims (args.pad)
// args.row_bytes_in  = full encoded row stride (turbo + tail)
// args.row_bytes_out = decoded half-row stride in bytes
kernel void kernel_ds4_turbo_kv_rope_tail_decode(
        constant ds4_metal_args_turbo & args,
        device const uchar            * src,
        device       half             * dst,
        uint  row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]]) {
    if ((int)row >= args.n_rows) return;
    const int head_dim = args.head_dim;
    const int n_rot    = args.pad;
    if (head_dim <= 0 || n_rot <= 0 || n_rot > head_dim) return;
    if ((int)tid >= n_rot) return;

    const uint src_stride = (uint)args.row_bytes_in;
    const uint dst_stride = (uint)(args.row_bytes_out / sizeof(half));
    const uint tail_off   = src_stride - (uint)n_rot * (uint)sizeof(half);

    device const uchar * tail = src + (uint)row * src_stride + tail_off;
    const ushort lo = (ushort)tail[(uint)tid * 2 + 0];
    const ushort hi = (ushort)tail[(uint)tid * 2 + 1];
    const ushort h  = lo | (hi << 8);

    const uint nope = (uint)(head_dim - n_rot);
    dst[(uint)row * dst_stride + nope + (uint)tid] = as_type<half>(h);
}

// Standalone WHT-128 helper for unit testing the transform in
// isolation. args.bits = 0 -> forward (signs1 -> FWHT -> signs2).
// args.bits = 1 -> inverse (signs2 -> FWHT -> signs1). The two
// directions are bit-identical inverses on the CPU side.
kernel void kernel_ds4_turbo_wht_inplace(
        constant ds4_metal_args_turbo & args,
        device       float            * data,
        threadgroup  float            * rot [[threadgroup(0)]],
        uint  row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]]) {
    const int head_dim = args.head_dim;
    if (head_dim != TURBO_ROT_D) return;

    device float * row_data = data + (uint)row * TURBO_ROT_D;
    rot[tid] = row_data[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (args.bits == 0) {
        rot[tid] *= turbo_wht_signs1[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        turbo_fwht_128_tg(rot, tid);
        rot[tid] *= turbo_wht_signs2[tid];
    } else {
        rot[tid] *= turbo_wht_signs2[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        turbo_fwht_128_tg(rot, tid);
        rot[tid] *= turbo_wht_signs1[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    row_data[tid] = rot[tid];
}
