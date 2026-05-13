#ifndef DS4_TURBO_H
#define DS4_TURBO_H

#include <stddef.h>
#include <stdint.h>

/* TurboQuant KV cache compression.
 *
 * 3-bit/4-bit quantization with Walsh-Hadamard rotation that gaussianizes
 * the per-row distribution before codebook lookup. Yields ~4.6x compression
 * vs FP16 at 3-bit with PPL within ~1% of FP8 round-trip.
 *
 * Operates on head_dim = 512 rows (DeepSeek V4 Flash). The rotation block
 * is 128 floats; we apply it four times per row.
 */

void   ds4_turbo_init(void);                          /* lazy init: WHT signs, QJL signs */

size_t ds4_turbo_row_bytes(int head_dim, int bits);   /* bytes per encoded row */

void   ds4_turbo_quantize_row(const float *src, void *dst,
                              int head_dim, int bits);
void   ds4_turbo_dequantize_row(const void *src, float *dst,
                                int head_dim, int bits);

/* Self test: encodes random Gaussian rows, checks cosine similarity > 0.99
 * for both 3-bit and 4-bit. Returns 0 on success, non-zero on failure. */
int    ds4_turbo_self_test(void);

/* Accessor for the runtime turbo KV bits configuration (0 = disabled, else
 * 3 or 4). Set by ds4.c at engine startup from $DS4_TURBO_KV_BITS so the
 * Metal backend can decide which KV store/read path to dispatch without
 * a static link into ds4.c internals. */
int    ds4_turbo_kv_bits_get(void);
void   ds4_turbo_kv_bits_set(int bits);

/* Block layout (per 32 floats):
 *   3-bit: half norm + 8B qs + 4B signs  = 14 B
 *   4-bit: half norm + 8B qs + 4B signs + 4B qjl_sign = 18 B
 */

#endif
