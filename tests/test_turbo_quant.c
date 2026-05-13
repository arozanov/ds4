/* Standalone smoke test for ds4_turbo.
 *
 * Compile and run:
 *   clang -O2 ../ds4_turbo.c test_turbo_quant.c -o test_turbo && ./test_turbo
 *
 * Exits 0 on success, non-zero on failure. */

#include "../ds4_turbo.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int check_roundtrip(int bits) {
    const int head_dim = 512;
    const int n_rows   = 8;

    float *in  = (float *)malloc(sizeof(float) * head_dim * n_rows);
    float *out = (float *)malloc(sizeof(float) * head_dim * n_rows);
    if (!in || !out) {
        free(in); free(out);
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    /* Box-Muller from a fixed seed so results are repeatable. */
    unsigned long long s = 0xBADF00DULL;
    for (int i = 0; i < head_dim * n_rows; i += 2) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        unsigned long long r1 = s;
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        unsigned long long r2 = s;
        double u1 = ((double)(r1 >> 11)) * (1.0 / 9007199254740992.0);
        double u2 = ((double)(r2 >> 11)) * (1.0 / 9007199254740992.0);
        if (u1 < 1e-15) u1 = 1e-15;
        double rr = sqrt(-2.0 * log(u1));
        double tt = 6.283185307179586 * u2;
        in[i] = (float)(rr * cos(tt));
        if (i + 1 < head_dim * n_rows) in[i + 1] = (float)(rr * sin(tt));
    }

    size_t row_bytes = ds4_turbo_row_bytes(head_dim, bits);
    if (!row_bytes) {
        fprintf(stderr, "bad row_bytes for bits=%d\n", bits);
        free(in); free(out);
        return 2;
    }

    unsigned char *enc = (unsigned char *)malloc(row_bytes * n_rows);
    if (!enc) {
        free(in); free(out);
        fprintf(stderr, "alloc enc failed\n");
        return 3;
    }

    for (int r = 0; r < n_rows; r++) {
        ds4_turbo_quantize_row(in + r * head_dim,
                               enc + (size_t)r * row_bytes,
                               head_dim, bits);
        ds4_turbo_dequantize_row(enc + (size_t)r * row_bytes,
                                 out + r * head_dim,
                                 head_dim, bits);
    }

    double dot = 0.0, na = 0.0, nb = 0.0, mse = 0.0;
    for (int i = 0; i < head_dim * n_rows; i++) {
        dot += (double)in[i] * (double)out[i];
        na  += (double)in[i] * (double)in[i];
        nb  += (double)out[i] * (double)out[i];
        double d = (double)in[i] - (double)out[i];
        mse += d * d;
    }
    double cos_sim = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    double nmse    = mse / (na + 1e-30);
    printf("bits=%d  row_bytes=%zu  cos_sim=%.6f  nmse=%.6f\n",
           bits, row_bytes, cos_sim, nmse);

    free(in);
    free(out);
    free(enc);
    /* 3-bit Lloyd-Max is bounded at SNR ~14.6 dB -> cos_sim ~0.98.
     * 4-bit must clear 0.99 to confirm the codebook is wired right. */
    const double thr = (bits == 3) ? 0.98 : 0.99;
    return (cos_sim > thr) ? 0 : 100;
}

int main(void) {
    ds4_turbo_init();
    int rc3 = check_roundtrip(3);
    int rc4 = check_roundtrip(4);
    int rc_self = ds4_turbo_self_test();
    printf("self_test=%d\n", rc_self);
    if (rc3 || rc4 || rc_self) {
        fprintf(stderr, "FAIL: rc3=%d rc4=%d rc_self=%d\n", rc3, rc4, rc_self);
        return 1;
    }
    printf("OK\n");
    return 0;
}
