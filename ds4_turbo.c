/* TurboQuant KV cache compression for DeepSeek V4.
 *
 * Two formats, both intended for head_dim = 512 rows:
 *   - 3-bit: per-block L2 norm (fp16) + 3-bit codebook index per value.
 *   - 4-bit: per-block L2 norm (fp16) + 4-bit codebook index per value.
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
    turbo_kv_bits_cached = (bits == 3 || bits == 4) ? bits : 0;
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

size_t ds4_turbo_row_bytes(int head_dim, int bits) {
    if (head_dim <= 0 || (head_dim % TURBO_BLOCK) != 0) return 0;
    const int nb = head_dim / TURBO_BLOCK;
    if (bits == 3) return (size_t)nb * 14u;
    if (bits == 4) return (size_t)nb * 18u;
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
    if (bits != 3 && bits != 4) return;

    const int n_rot = head_dim / TURBO_ROT_D;
    const size_t blk_bytes = (bits == 3) ? 14u : 18u;

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
            if (bits == 3) {
                for (int i = 0; i < TURBO_BLOCK; i++) {
                    idx[i] = (uint8_t)turbo_nearest3(blk[i] * scale);
                }
            } else {
                for (int i = 0; i < TURBO_BLOCK; i++) {
                    idx[i] = (uint8_t)turbo_nearest4(blk[i] * scale);
                }
            }

            uint16_t n16 = turbo_f32_to_f16(norm);
            blk_out[0] = (uint8_t)(n16 & 0xffu);
            blk_out[1] = (uint8_t)((n16 >> 8) & 0xffu);
            if (bits == 3) {
                turbo_pack_block3(idx, blk_out + 2, blk_out + 10);
            } else {
                turbo_pack_block4(idx, blk_out + 2, blk_out + 10, blk_out + 14);
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
    if (bits != 3 && bits != 4) return;

    const int n_rot = head_dim / TURBO_ROT_D;
    const size_t blk_bytes = (bits == 3) ? 14u : 18u;
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
            if (bits == 3) {
                turbo_unpack_block3(blk_in + 2, blk_in + 10, idx);
                for (int i = 0; i < TURBO_BLOCK; i++) {
                    rot[b * TURBO_BLOCK + i] = turbo_cb3[idx[i]] * inv_scale;
                }
            } else {
                turbo_unpack_block4(blk_in + 2, blk_in + 10, blk_in + 14, idx);
                for (int i = 0; i < TURBO_BLOCK; i++) {
                    rot[b * TURBO_BLOCK + i] = turbo_cb4[idx[i]] * inv_scale;
                }
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
    for (int bits = 3; bits <= 4; bits++) {
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
         * 4-bit gets to SNR ~20 dB and must clear 0.99. */
        const double thr = (bits == 3) ? 0.98 : 0.99;
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
