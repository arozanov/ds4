/* TurboQuant KV cache compression for DeepSeek V4.
 *
 * Four formats, all intended for head_dim = 512 rows:
 *   - 3-bit: per-block L2 norm (fp16) + 3-bit codebook index per value.
 *   - 4-bit: per-block L2 norm (fp16) + 4-bit codebook index per value.
 *   - 6-bit: per-block L2 norm (fp16) + 6-bit codebook index per value.
 *   - 8-bit: per-block L2 norm (fp16) + 8-bit codebook index per value.
 *
 * Pre-processing rotates each 128-float chunk via a normalized fast
 * Walsh-Hadamard transform sandwiched between two random sign masks. WHT
 * is its own inverse, so the masks (applied in reversed order) recover
 * the original chunk on dequant. The transform gaussianizes the per-row
 * distribution so a codebook trained on N(0, 1) fits well irrespective
 * of the input statistics. This is the QJL trick: the random sign masks
 * make the transform behave like a random orthonormal projection.
 *
 * Block sizing:
 *   - Rotation block: 128 floats (applied 4 times per row of 512).
 *   - Quantization block: 32 floats. 16 blocks per row of 512.
 *
 * The rotation block is larger than the quantization block on purpose:
 * gaussianization wants O(128) values for a clean tail, while 32 is the
 * smallest block that keeps the per-block norm overhead under 2 bits per
 * value (fp16 norm / 32 = 0.5 bit per value).
 *
 * Encoded layout per 32-float block:
 *   - 3-bit: 2B norm + 8B qs (low 2 bits of each index) + 4B signs (high
 *     bit). 14 bytes.
 *   - 4-bit: 3-bit layout above plus 4 more bytes of "qjl_sign" holding
 *     the high bit of a 4-bit index. 18 bytes. The qs / signs / qjl_sign
 *     split is inherited from the QJL-flavored variant; here we use it
 *     just as a bit-sliced 4-bit codebook index.
 *   - 6-bit: 2B norm + 24B qs (32 values * 6 bits = 192 bits, packed
 *     little-endian). 26 bytes. No separate sign bits; the codebook is
 *     fully signed (Lloyd-Max for N(0,1), symmetric over zero).
 *   - 8-bit: 2B norm + 32B qs (one byte per index). 34 bytes. Same
 *     rationale: signed codebook, no auxiliary bits.
 */

#include "ds4_turbo.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TURBO_ROT_D    128         /* WHT block size */
#define TURBO_BLOCK    32          /* quantization block size */
#define TURBO_SEED_ROT 42u

/* Lloyd-Max optimal centroids for N(0, 1), 8 levels (3-bit codebook). */
static const float turbo_cb3[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f,
};

/* Midpoints between adjacent centroids, used by the encoder to pick the
 * nearest centroid by linear scan (small enough that a binary search is
 * not worth the code). */
static const float turbo_cb3_mid[7] = {
    -1.7479f, -1.0499f, -0.5005f, 0.0000f,
     0.5005f,  1.0499f,  1.7479f,
};

/* Lloyd-Max optimal centroids for N(0, 1), 16 levels (4-bit codebook).
 * Distortion D ~ 0.00946 -> SNR ~ 20.2 dB, well above the 3-bit ~14.6 dB. */
static const float turbo_cb4[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f,
};

static const float turbo_cb4_mid[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f,
    -0.7995f, -0.5224f, -0.2582f,  0.0000f,
     0.2582f,  0.5224f,  0.7995f,  1.0993f,
     1.4371f,  1.8435f,  2.4008f,
};

/* Lloyd-Max optimal centroids for N(0, 1), 64 levels (6-bit codebook).
 * Distortion D ~ 0.00239 -> SNR ~ 26.2 dB. Codebook is symmetric and
 * spans both signs, so no separate sign bit is required. */
static const float turbo_cb6[64] = {
    -3.605999f, -3.085722f, -2.751095f, -2.496953f, -2.288778f, -2.110705f, -1.954065f, -1.813578f,
    -1.685776f, -1.568258f, -1.459279f, -1.357532f, -1.262008f, -1.171905f, -1.086573f, -1.005472f,
    -0.928148f, -0.854207f, -0.783309f, -0.715146f, -0.649445f, -0.585951f, -0.524433f, -0.464669f,
    -0.406453f, -0.349584f, -0.293873f, -0.239131f, -0.185179f, -0.131837f, -0.078929f, -0.026281f,
     0.026281f,  0.078929f,  0.131837f,  0.185179f,  0.239131f,  0.293873f,  0.349584f,  0.406453f,
     0.464669f,  0.524433f,  0.585951f,  0.649445f,  0.715146f,  0.783309f,  0.854207f,  0.928148f,
     1.005472f,  1.086573f,  1.171905f,  1.262008f,  1.357532f,  1.459279f,  1.568258f,  1.685776f,
     1.813578f,  1.954065f,  2.110705f,  2.288778f,  2.496953f,  2.751095f,  3.085722f,  3.605999f,
};

static const float turbo_cb6_mid[63] = {
    -3.345860f, -2.918408f, -2.624024f, -2.392865f, -2.199742f, -2.032385f, -1.883821f, -1.749677f,
    -1.627017f, -1.513768f, -1.408405f, -1.309770f, -1.216957f, -1.129239f, -1.046022f, -0.966810f,
    -0.891178f, -0.818758f, -0.749228f, -0.682295f, -0.617698f, -0.555192f, -0.494551f, -0.435561f,
    -0.378018f, -0.321728f, -0.266502f, -0.212155f, -0.158508f, -0.105383f, -0.052605f,  0.000000f,
     0.052605f,  0.105383f,  0.158508f,  0.212155f,  0.266502f,  0.321728f,  0.378018f,  0.435561f,
     0.494551f,  0.555192f,  0.617698f,  0.682295f,  0.749228f,  0.818758f,  0.891178f,  0.966810f,
     1.046022f,  1.129239f,  1.216957f,  1.309770f,  1.408405f,  1.513768f,  1.627017f,  1.749677f,
     1.883821f,  2.032385f,  2.199742f,  2.392865f,  2.624024f,  2.918408f,  3.345860f,
};

/* Lloyd-Max optimal centroids for N(0, 1), 256 levels (8-bit codebook).
 * Distortion D ~ 0.000597 -> SNR ~ 32.2 dB. */
static const float turbo_cb8[256] = {
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

static const float turbo_cb8_mid[255] = {
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

/* Lazy-initialized random sign masks. Two 128-element +/-1 masks
 * (applied before and after the WHT) turn the deterministic transform
 * into a random-feeling orthonormal map without storing a dense
 * 128x128 matrix. */
static float turbo_wht_signs1[TURBO_ROT_D];
static float turbo_wht_signs2[TURBO_ROT_D];
static int   turbo_initialized = 0;

/* Deterministic 64-bit LCG. We only use it to derive sign bits, so
 * statistical quality beyond "balanced" is not required. */
static uint64_t turbo_lcg(uint64_t *state) {
    *state = (*state) * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state;
}

static void turbo_fill_signs(uint64_t seed, float *out, int n) {
    uint64_t s = seed;
    for (int i = 0; i < n; i++) {
        /* Take a high bit so we drift across the period. */
        out[i] = ((turbo_lcg(&s) >> 33) & 1u) ? 1.0f : -1.0f;
    }
}

void ds4_turbo_init(void) {
    if (turbo_initialized) return;
    turbo_fill_signs(TURBO_SEED_ROT,      turbo_wht_signs1, TURBO_ROT_D);
    turbo_fill_signs(TURBO_SEED_ROT + 1u, turbo_wht_signs2, TURBO_ROT_D);
    turbo_initialized = 1;
}

/* The KV bits selector is owned by ds4.c (which reads $DS4_TURBO_KV_BITS at
 * startup), but kept here so backends without ds4.c symbol access can still
 * branch on it. ds4.c calls _set() once; everyone else calls _get(). */
static int turbo_kv_bits_cached;

int ds4_turbo_kv_bits_get(void) {
    return turbo_kv_bits_cached;
}

void ds4_turbo_kv_bits_set(int bits) {
    turbo_kv_bits_cached =
        (bits == 3 || bits == 4 || bits == 6 || bits == 8) ? bits : 0;
}

/* Portable IEEE 754 binary16 conversion. We do not require the host
 * fp16 instructions; the ds4 main path has its own NEON-accelerated
 * versions, but this file keeps the dependency surface trivial. */
static uint16_t turbo_f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }
    uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (h & 1u))) h++;
    return (uint16_t)h;
}

static float turbo_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* In-place normalized fast Walsh-Hadamard transform on 128 floats.
 * The transform is self-inverse modulo a 1/sqrt(128) normalization, so
 * applying twice (with the same sign masks) recovers the input. */
static void turbo_fwht_128(float *x) {
    for (int h = 1; h < TURBO_ROT_D; h *= 2) {
        for (int i = 0; i < TURBO_ROT_D; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    /* 1/sqrt(128) keeps the transform orthonormal. */
    const float inv = 0.08838834764831845f;
    for (int i = 0; i < TURBO_ROT_D; i++) x[i] *= inv;
}

/* Forward rotation: signs1 -> FWHT -> signs2. The two sign masks turn
 * the deterministic WHT into a random-feeling orthonormal map without
 * having to store a dense 128x128 matrix. */
static void turbo_rotate_forward(float *x, const float *s1, const float *s2) {
    for (int i = 0; i < TURBO_ROT_D; i++) x[i] *= s1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < TURBO_ROT_D; i++) x[i] *= s2[i];
}

/* Inverse rotation. WHT is self-inverse, so we just undo the sign
 * masks in reversed order. */
static void turbo_rotate_inverse(float *x, const float *s1, const float *s2) {
    for (int i = 0; i < TURBO_ROT_D; i++) x[i] *= s2[i];
    turbo_fwht_128(x);
    for (int i = 0; i < TURBO_ROT_D; i++) x[i] *= s1[i];
}

/* Nearest centroid by midpoint comparison. Codebooks are sorted, so a
 * linear scan over 7 (resp. 15) thresholds is shorter than the
 * equivalent binary search and easier to autovectorize. */
static int turbo_nearest3(float v) {
    int i = 0;
    while (i < 7 && v >= turbo_cb3_mid[i]) i++;
    return i;
}

static int turbo_nearest4(float v) {
    int i = 0;
    while (i < 15 && v >= turbo_cb4_mid[i]) i++;
    return i;
}

/* 6/8-bit codebooks span both signs so the encoder runs the full
 * threshold scan over all centroids. The lists are sorted, so a linear
 * scan over 63 (resp. 255) midpoints is still vectorizable; binary
 * search is a wash at this size and would complicate the GPU mirror. */
static int turbo_nearest6(float v) {
    int i = 0;
    while (i < 63 && v >= turbo_cb6_mid[i]) i++;
    return i;
}

static int turbo_nearest8(float v) {
    int i = 0;
    while (i < 255 && v >= turbo_cb8_mid[i]) i++;
    return i;
}

size_t ds4_turbo_row_bytes(int head_dim, int bits) {
    if (head_dim <= 0 || (head_dim % TURBO_BLOCK) != 0) return 0;
    const int nb = head_dim / TURBO_BLOCK;
    /* 2B norm + per-block index payload. The 3/4-bit formats split the
     * codebook index into qs/signs/qjl slots; 6/8-bit pack the full
     * signed index inline (no sign bits, codebook is symmetric). */
    if (bits == 3) return (size_t)nb * 14u;
    if (bits == 4) return (size_t)nb * 18u;
    if (bits == 6) return (size_t)nb * 26u; /* 2B norm + 24B qs */
    if (bits == 8) return (size_t)nb * 34u; /* 2B norm + 32B qs */
    return 0;
}

/* Per-block packing helpers. Indices are bit-sliced so that the 3-bit
 * format is a strict prefix of the 4-bit format on disk:
 *
 *   qs        (8B): bits 0-1 of each index, 4 indices per byte.
 *   signs     (4B): bit 2 of each index, 8 indices per byte.
 *   qjl_sign  (4B): bit 3 (4-bit format only), 8 indices per byte.
 *
 * The "signs" / "qjl_sign" names are inherited from the QJL-flavored
 * variant; here they are just slots for the upper bits of the codebook
 * index. */

static void turbo_pack_block3(const uint8_t *idx, uint8_t *qs, uint8_t *signs) {
    memset(qs,    0, 8);
    memset(signs, 0, 4);
    for (int i = 0; i < TURBO_BLOCK; i++) {
        const uint8_t v = idx[i] & 0x7u;
        qs[i >> 2]    |= (uint8_t)((v & 0x3u) << ((i & 3) * 2));
        signs[i >> 3] |= (uint8_t)(((v >> 2) & 0x1u) << (i & 7));
    }
}

static void turbo_unpack_block3(const uint8_t *qs, const uint8_t *signs,
                                uint8_t *idx) {
    for (int i = 0; i < TURBO_BLOCK; i++) {
        const uint8_t lo = (qs[i >> 2] >> ((i & 3) * 2)) & 0x3u;
        const uint8_t hi = (signs[i >> 3] >> (i & 7)) & 0x1u;
        idx[i] = (uint8_t)(lo | (hi << 2));
    }
}

static void turbo_pack_block4(const uint8_t *idx, uint8_t *qs,
                              uint8_t *signs, uint8_t *qjl) {
    memset(qs,    0, 8);
    memset(signs, 0, 4);
    memset(qjl,   0, 4);
    for (int i = 0; i < TURBO_BLOCK; i++) {
        const uint8_t v = idx[i] & 0xfu;
        qs[i >> 2]    |= (uint8_t)((v & 0x3u) << ((i & 3) * 2));
        signs[i >> 3] |= (uint8_t)(((v >> 2) & 0x1u) << (i & 7));
        qjl[i >> 3]   |= (uint8_t)(((v >> 3) & 0x1u) << (i & 7));
    }
}

static void turbo_unpack_block4(const uint8_t *qs, const uint8_t *signs,
                                const uint8_t *qjl, uint8_t *idx) {
    for (int i = 0; i < TURBO_BLOCK; i++) {
        const uint8_t b0 = (qs[i >> 2] >> ((i & 3) * 2)) & 0x3u;
        const uint8_t b1 = (signs[i >> 3] >> (i & 7)) & 0x1u;
        const uint8_t b2 = (qjl[i >> 3]   >> (i & 7)) & 0x1u;
        idx[i] = (uint8_t)(b0 | (b1 << 2) | (b2 << 3));
    }
}

/* 6-bit packing: 32 values * 6 bits = 192 bits = 24 bytes.
 * Pack 4 indices (24 bits) into 3 bytes, little-endian. The 32 indices
 * therefore split into 8 groups of 4, each occupying 3 bytes.
 *
 * For group g of 4 indices (a, b, c, d), the 24-bit word is
 *   word = a | (b << 6) | (c << 12) | (d << 18)
 * stored as little-endian bytes (qs[3g+0..3g+2]). */
static void turbo_pack_block6(const uint8_t *idx, uint8_t *qs) {
    for (int g = 0; g < TURBO_BLOCK / 4; g++) {
        const uint32_t a = idx[4*g + 0] & 0x3fu;
        const uint32_t b = idx[4*g + 1] & 0x3fu;
        const uint32_t c = idx[4*g + 2] & 0x3fu;
        const uint32_t d = idx[4*g + 3] & 0x3fu;
        const uint32_t w = a | (b << 6) | (c << 12) | (d << 18);
        qs[3*g + 0] = (uint8_t)( w        & 0xffu);
        qs[3*g + 1] = (uint8_t)((w >>  8) & 0xffu);
        qs[3*g + 2] = (uint8_t)((w >> 16) & 0xffu);
    }
}

static void turbo_unpack_block6(const uint8_t *qs, uint8_t *idx) {
    for (int g = 0; g < TURBO_BLOCK / 4; g++) {
        const uint32_t w =
            (uint32_t)qs[3*g + 0]        |
            ((uint32_t)qs[3*g + 1] <<  8) |
            ((uint32_t)qs[3*g + 2] << 16);
        idx[4*g + 0] = (uint8_t)( w        & 0x3fu);
        idx[4*g + 1] = (uint8_t)((w >>  6) & 0x3fu);
        idx[4*g + 2] = (uint8_t)((w >> 12) & 0x3fu);
        idx[4*g + 3] = (uint8_t)((w >> 18) & 0x3fu);
    }
}

/* 8-bit packing: one byte per index. Trivial. */
static void turbo_pack_block8(const uint8_t *idx, uint8_t *qs) {
    memcpy(qs, idx, TURBO_BLOCK);
}

static void turbo_unpack_block8(const uint8_t *qs, uint8_t *idx) {
    memcpy(idx, qs, TURBO_BLOCK);
}

/* Encode one head_dim-long row. Layout per row:
 *   - head_dim / 128 rotation groups, each rotated independently with
 *     the same shared sign masks;
 *   - head_dim / 32 quantization blocks; each block stores fp16 norm
 *     plus a 3- or 4-bit codebook index for every value.
 *
 * The norm is the post-rotation block L2; we divide by it before
 * codebook lookup so the codebook can be trained once on N(0, 1)
 * irrespective of the input scale. */
void ds4_turbo_quantize_row(const float *src, void *dst,
                            int head_dim, int bits) {
    if (!turbo_initialized) ds4_turbo_init();
    if (head_dim <= 0 || (head_dim % TURBO_ROT_D) != 0) return;
    if (bits != 3 && bits != 4 && bits != 6 && bits != 8) return;

    const int n_rot = head_dim / TURBO_ROT_D;
    size_t blk_bytes;
    switch (bits) {
        case 3: blk_bytes = 14u; break;
        case 4: blk_bytes = 18u; break;
        case 6: blk_bytes = 26u; break;
        default: blk_bytes = 34u; break; /* bits == 8 */
    }

    float rot[TURBO_ROT_D];
    uint8_t *out = (uint8_t *)dst;

    for (int g = 0; g < n_rot; g++) {
        memcpy(rot, src + g * TURBO_ROT_D, sizeof(rot));
        turbo_rotate_forward(rot, turbo_wht_signs1, turbo_wht_signs2);

        for (int b = 0; b < TURBO_ROT_D / TURBO_BLOCK; b++) {
            const float *blk = rot + b * TURBO_BLOCK;
            const int block_idx = g * (TURBO_ROT_D / TURBO_BLOCK) + b;
            uint8_t *blk_out = out + (size_t)block_idx * blk_bytes;

            float norm_sq = 0.0f;
            for (int i = 0; i < TURBO_BLOCK; i++) norm_sq += blk[i] * blk[i];
            float norm = sqrtf(norm_sq);
            /* After rotation, the block has L2 ~= sqrt(32) * sigma.
             * Dividing by norm/sqrt(32) puts each component on the
             * N(0, 1) scale the codebook expects. */
            float scale = (norm > 1e-12f) ? (sqrtf((float)TURBO_BLOCK) / norm) : 0.0f;

            uint8_t idx[TURBO_BLOCK];
            switch (bits) {
                case 3:
                    for (int i = 0; i < TURBO_BLOCK; i++) {
                        idx[i] = (uint8_t)turbo_nearest3(blk[i] * scale);
                    }
                    break;
                case 4:
                    for (int i = 0; i < TURBO_BLOCK; i++) {
                        idx[i] = (uint8_t)turbo_nearest4(blk[i] * scale);
                    }
                    break;
                case 6:
                    for (int i = 0; i < TURBO_BLOCK; i++) {
                        idx[i] = (uint8_t)turbo_nearest6(blk[i] * scale);
                    }
                    break;
                default: /* 8 */
                    for (int i = 0; i < TURBO_BLOCK; i++) {
                        idx[i] = (uint8_t)turbo_nearest8(blk[i] * scale);
                    }
                    break;
            }

            uint16_t n16 = turbo_f32_to_f16(norm);
            blk_out[0] = (uint8_t)(n16 & 0xffu);
            blk_out[1] = (uint8_t)((n16 >> 8) & 0xffu);
            switch (bits) {
                case 3:
                    turbo_pack_block3(idx, blk_out + 2, blk_out + 10);
                    break;
                case 4:
                    turbo_pack_block4(idx, blk_out + 2, blk_out + 10, blk_out + 14);
                    break;
                case 6:
                    turbo_pack_block6(idx, blk_out + 2);
                    break;
                default: /* 8 */
                    turbo_pack_block8(idx, blk_out + 2);
                    break;
            }
        }
    }
}

/* Decode one head_dim-long row. Mirror of the encoder: dequant each
 * 32-float block back into the rotation domain, then apply the inverse
 * rotation per 128-float group. */
void ds4_turbo_dequantize_row(const void *src, float *dst,
                              int head_dim, int bits) {
    if (!turbo_initialized) ds4_turbo_init();
    if (head_dim <= 0 || (head_dim % TURBO_ROT_D) != 0) return;
    if (bits != 3 && bits != 4 && bits != 6 && bits != 8) return;

    const int n_rot = head_dim / TURBO_ROT_D;
    size_t blk_bytes;
    switch (bits) {
        case 3: blk_bytes = 14u; break;
        case 4: blk_bytes = 18u; break;
        case 6: blk_bytes = 26u; break;
        default: blk_bytes = 34u; break;
    }
    const uint8_t *in = (const uint8_t *)src;

    float rot[TURBO_ROT_D];

    for (int g = 0; g < n_rot; g++) {
        for (int b = 0; b < TURBO_ROT_D / TURBO_BLOCK; b++) {
            const int block_idx = g * (TURBO_ROT_D / TURBO_BLOCK) + b;
            const uint8_t *blk_in = in + (size_t)block_idx * blk_bytes;
            uint16_t n16 = (uint16_t)blk_in[0] | ((uint16_t)blk_in[1] << 8);
            float norm = turbo_f16_to_f32(n16);
            float inv_scale = norm * (1.0f / sqrtf((float)TURBO_BLOCK));

            uint8_t idx[TURBO_BLOCK];
            switch (bits) {
                case 3:
                    turbo_unpack_block3(blk_in + 2, blk_in + 10, idx);
                    for (int i = 0; i < TURBO_BLOCK; i++) {
                        rot[b * TURBO_BLOCK + i] = turbo_cb3[idx[i]] * inv_scale;
                    }
                    break;
                case 4:
                    turbo_unpack_block4(blk_in + 2, blk_in + 10, blk_in + 14, idx);
                    for (int i = 0; i < TURBO_BLOCK; i++) {
                        rot[b * TURBO_BLOCK + i] = turbo_cb4[idx[i]] * inv_scale;
                    }
                    break;
                case 6:
                    turbo_unpack_block6(blk_in + 2, idx);
                    for (int i = 0; i < TURBO_BLOCK; i++) {
                        rot[b * TURBO_BLOCK + i] = turbo_cb6[idx[i]] * inv_scale;
                    }
                    break;
                default: /* 8 */
                    turbo_unpack_block8(blk_in + 2, idx);
                    for (int i = 0; i < TURBO_BLOCK; i++) {
                        rot[b * TURBO_BLOCK + i] = turbo_cb8[idx[i]] * inv_scale;
                    }
                    break;
            }
        }

        turbo_rotate_inverse(rot, turbo_wht_signs1, turbo_wht_signs2);
        memcpy(dst + g * TURBO_ROT_D, rot, sizeof(rot));
    }
}

/* Round-trip random data and check cosine similarity. Returns 0 on
 * success. Kept lightweight so it can run from a unit test or, if
 * a developer wires it up, from the bench binary. */
int ds4_turbo_self_test(void) {
    const int head_dim = 512;
    const int n_rows   = 8;

    ds4_turbo_init();

    float *in  = (float *)malloc(sizeof(float) * head_dim * n_rows);
    float *out = (float *)malloc(sizeof(float) * head_dim * n_rows);
    if (!in || !out) {
        free(in); free(out);
        return 1;
    }

    /* Box-Muller from a fixed-seed LCG so the self test is repeatable. */
    uint64_t s = 0xC0FFEEULL;
    for (int i = 0; i < head_dim * n_rows; i += 2) {
        const uint64_t r1 = turbo_lcg(&s);
        const uint64_t r2 = turbo_lcg(&s);
        double u1 = ((double)(r1 >> 11)) * (1.0 / 9007199254740992.0);
        double u2 = ((double)(r2 >> 11)) * (1.0 / 9007199254740992.0);
        if (u1 < 1e-15) u1 = 1e-15;
        const double r = sqrt(-2.0 * log(u1));
        const double t = 6.283185307179586 * u2;
        in[i]     = (float)(r * cos(t));
        if (i + 1 < head_dim * n_rows) in[i + 1] = (float)(r * sin(t));
    }

    int rc = 0;
    static const int bit_widths[] = {3, 4, 6, 8};
    for (size_t bi = 0; bi < sizeof(bit_widths) / sizeof(bit_widths[0]); bi++) {
        const int bits = bit_widths[bi];
        const size_t row_bytes = ds4_turbo_row_bytes(head_dim, bits);
        uint8_t *enc = (uint8_t *)malloc(row_bytes * n_rows);
        if (!enc) { rc = 2; break; }

        for (int r = 0; r < n_rows; r++) {
            ds4_turbo_quantize_row(in + r * head_dim,
                                   enc + (size_t)r * row_bytes,
                                   head_dim, bits);
            ds4_turbo_dequantize_row(enc + (size_t)r * row_bytes,
                                     out + r * head_dim,
                                     head_dim, bits);
        }

        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < head_dim * n_rows; i++) {
            dot += (double)in[i] * (double)out[i];
            na  += (double)in[i] * (double)in[i];
            nb  += (double)out[i] * (double)out[i];
        }
        const double cos_sim = dot / (sqrt(na) * sqrt(nb) + 1e-30);
        /* 3-bit Lloyd-Max sits at SNR ~14.6 dB so we accept >= 0.98;
         * 4-bit gets to SNR ~20 dB and must clear 0.99;
         * 6-bit (SNR ~26 dB) clears 0.998; 8-bit (SNR ~32 dB) clears 0.9995. */
        double thr;
        switch (bits) {
            case 3:  thr = 0.98;   break;
            case 4:  thr = 0.99;   break;
            case 6:  thr = 0.998;  break;
            default: thr = 0.9995; break;
        }
        if (!(cos_sim > thr)) {
            rc = 10 + bits;
            free(enc);
            break;
        }
        free(enc);
    }

    free(in);
    free(out);
    return rc;
}
