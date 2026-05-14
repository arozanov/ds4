/* Standalone GPU smoke test for ds4_turbo Metal kernels.
 *
 * Compile and run:
 *   cd /Users/antonrozanov/Projects/turboquant-money/ds4
 *   clang -O2 -fobjc-arc \
 *       -framework Foundation -framework Metal \
 *       ds4_turbo.c tests/test_turbo_metal.m -o tests/test_turbo_metal \
 *       && ./tests/test_turbo_metal
 *
 * Loads metal/turbo.metal into a private MTLLibrary (no dependency on
 * ds4_metal.m), runs the encoder kernel against random Gaussian rows,
 * decodes back on CPU, then runs the GPU decoder on the GPU-produced
 * bytes. Both round-trips must clear cos_sim > 0.95.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../ds4_turbo.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEAD_DIM 512
#define N_ROWS   8

typedef struct {
    int32_t  head_dim;
    int32_t  bits;
    int32_t  n_rows;
    int32_t  pad;
    uint64_t row_bytes_in;
    uint64_t row_bytes_out;
} turbo_args_t;

/* Box-Muller from a fixed-seed LCG. Matches test_turbo_quant.c so the
 * GPU vs CPU comparison uses the same input distribution. */
static void fill_gaussian(float *buf, size_t n, unsigned long long seed) {
    unsigned long long s = seed;
    for (size_t i = 0; i < n; i += 2) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        unsigned long long r1 = s;
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        unsigned long long r2 = s;
        double u1 = ((double)(r1 >> 11)) * (1.0 / 9007199254740992.0);
        double u2 = ((double)(r2 >> 11)) * (1.0 / 9007199254740992.0);
        if (u1 < 1e-15) u1 = 1e-15;
        double rr = sqrt(-2.0 * log(u1));
        double tt = 6.283185307179586 * u2;
        buf[i] = (float)(rr * cos(tt));
        if (i + 1 < n) buf[i + 1] = (float)(rr * sin(tt));
    }
}

static double cosine_similarity(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-30);
}

static NSString *load_metal_source(void) {
    NSMutableString *src = [NSMutableString string];
    [src appendString:@"#include <metal_stdlib>\n"];
    [src appendString:@"using namespace metal;\n\n"];

    NSError *err = nil;
    NSString *path = @"metal/turbo.metal";
    NSString *body = [NSString stringWithContentsOfFile:path
                                               encoding:NSUTF8StringEncoding
                                                  error:&err];
    if (!body) {
        fprintf(stderr, "test_turbo_metal: cannot read %s: %s\n",
                [path UTF8String], [[err localizedDescription] UTF8String]);
        return nil;
    }
    [src appendString:body];
    return src;
}

int main(void) {
    @autoreleasepool {
        ds4_turbo_init();

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "test_turbo_metal: no Metal device available\n");
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            fprintf(stderr, "test_turbo_metal: cannot create command queue\n");
            return 1;
        }

        NSString *source = load_metal_source();
        if (!source) return 1;

        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [device newLibraryWithSource:source
                                                  options:opts
                                                    error:&err];
        if (!lib) {
            fprintf(stderr, "test_turbo_metal: compile failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            return 1;
        }
        fprintf(stdout, "metal library compiled\n");

        id<MTLFunction> fn_enc = [lib newFunctionWithName:@"kernel_ds4_turbo_quantize_row"];
        id<MTLFunction> fn_dec = [lib newFunctionWithName:@"kernel_ds4_turbo_dequantize_row"];
        id<MTLFunction> fn_bat = [lib newFunctionWithName:@"kernel_ds4_turbo_dequantize_batch"];
        id<MTLFunction> fn_wht = [lib newFunctionWithName:@"kernel_ds4_turbo_wht_inplace"];
        if (!fn_enc || !fn_dec || !fn_bat || !fn_wht) {
            fprintf(stderr, "test_turbo_metal: kernel lookup failed (enc=%p dec=%p bat=%p wht=%p)\n",
                    fn_enc, fn_dec, fn_bat, fn_wht);
            return 1;
        }
        id<MTLComputePipelineState> ps_enc = [device newComputePipelineStateWithFunction:fn_enc error:&err];
        id<MTLComputePipelineState> ps_dec = [device newComputePipelineStateWithFunction:fn_dec error:&err];
        id<MTLComputePipelineState> ps_bat = [device newComputePipelineStateWithFunction:fn_bat error:&err];
        id<MTLComputePipelineState> ps_wht = [device newComputePipelineStateWithFunction:fn_wht error:&err];
        if (!ps_enc || !ps_dec || !ps_bat || !ps_wht) {
            fprintf(stderr, "test_turbo_metal: pipeline create failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            return 1;
        }
        fprintf(stdout, "pipelines ready\n");

        const size_t row_floats = HEAD_DIM;
        const size_t input_bytes = sizeof(float) * row_floats * N_ROWS;

        float *input = (float *)malloc(input_bytes);
        if (!input) { fprintf(stderr, "alloc input failed\n"); return 1; }
        fill_gaussian(input, row_floats * N_ROWS, 0xBADF00DULL);

        int overall_rc = 0;

        static const int bit_widths[] = {3, 4, 6, 8};
        for (size_t bi = 0; bi < sizeof(bit_widths)/sizeof(bit_widths[0]); bi++) {
            const int bits = bit_widths[bi];
            const size_t row_bytes = ds4_turbo_row_bytes(HEAD_DIM, bits);
            if (!row_bytes) { fprintf(stderr, "bad row_bytes\n"); overall_rc = 2; break; }
            const size_t enc_bytes = row_bytes * N_ROWS;

            id<MTLBuffer> bufIn  = [device newBufferWithBytes:input length:input_bytes
                                                     options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufEnc = [device newBufferWithLength:enc_bytes
                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufOutF16 = [device newBufferWithLength:sizeof(uint16_t) * row_floats * N_ROWS
                                                         options:MTLResourceStorageModeShared];

            turbo_args_t args = (turbo_args_t){
                .head_dim      = HEAD_DIM,
                .bits          = bits,
                .n_rows        = N_ROWS,
                .pad           = 0,
                .row_bytes_in  = row_bytes,
                .row_bytes_out = sizeof(uint16_t) * row_floats,
            };
            id<MTLBuffer> bufArgs = [device newBufferWithBytes:&args length:sizeof(args)
                                                       options:MTLResourceStorageModeShared];

            // ----- GPU encode -----
            {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps_enc];
                [enc setBuffer:bufArgs offset:0 atIndex:0];
                [enc setBuffer:bufIn   offset:0 atIndex:1];
                [enc setBuffer:bufEnc  offset:0 atIndex:2];
                // Threadgroup mem: rot = 128 floats, scratch = 672 floats
                [enc setThreadgroupMemoryLength:sizeof(float) * 128 atIndex:0];
                [enc setThreadgroupMemoryLength:sizeof(float) * 672 atIndex:1];
                MTLSize tg = MTLSizeMake(128, 1, 1);
                MTLSize grid = MTLSizeMake(N_ROWS, 1, 1);
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.error) {
                    fprintf(stderr, "encode dispatch error: %s\n",
                            [[cb.error localizedDescription] UTF8String]);
                    overall_rc = 3; break;
                }
            }

            // ----- CPU decode of GPU-encoded bytes -----
            float *cpu_out = (float *)malloc(sizeof(float) * row_floats * N_ROWS);
            for (int r = 0; r < N_ROWS; r++) {
                ds4_turbo_dequantize_row((const uint8_t *)[bufEnc contents] + (size_t)r * row_bytes,
                                         cpu_out + (size_t)r * row_floats,
                                         HEAD_DIM, bits);
            }
            double cs_gpu_enc = cosine_similarity(input, cpu_out, row_floats * N_ROWS);
            fprintf(stdout, "bits=%d  GPU encode + CPU decode  cos_sim=%.6f\n", bits, cs_gpu_enc);

            // Also compare GPU encoded bytes vs CPU encoded bytes for
            // bit-identity — they should be byte-equal modulo any norm
            // rounding from sqrt() error.
            uint8_t *cpu_enc = (uint8_t *)malloc(enc_bytes);
            for (int r = 0; r < N_ROWS; r++) {
                ds4_turbo_quantize_row(input + (size_t)r * row_floats,
                                       cpu_enc + (size_t)r * row_bytes,
                                       HEAD_DIM, bits);
            }
            size_t diff = 0;
            const uint8_t *gpu_enc = (const uint8_t *)[bufEnc contents];
            for (size_t i = 0; i < enc_bytes; i++) {
                if (cpu_enc[i] != gpu_enc[i]) diff++;
            }
            fprintf(stdout, "bits=%d  GPU vs CPU encoded byte diff = %zu / %zu\n",
                    bits, diff, enc_bytes);

            // ----- GPU decode of GPU-encoded bytes -----
            {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> dec = [cb computeCommandEncoder];
                [dec setComputePipelineState:ps_dec];
                [dec setBuffer:bufArgs   offset:0 atIndex:0];
                [dec setBuffer:bufEnc    offset:0 atIndex:1];
                [dec setBuffer:bufOutF16 offset:0 atIndex:2];
                [dec setThreadgroupMemoryLength:sizeof(float) * 128 atIndex:0];
                [dec setThreadgroupMemoryLength:sizeof(float) *   4 atIndex:1];
                MTLSize tg = MTLSizeMake(128, 1, 1);
                MTLSize grid = MTLSizeMake(N_ROWS, 1, 1);
                [dec dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                [dec endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.error) {
                    fprintf(stderr, "decode dispatch error: %s\n",
                            [[cb.error localizedDescription] UTF8String]);
                    overall_rc = 4;
                    free(cpu_out); free(cpu_enc);
                    break;
                }
            }

            // Convert fp16 -> fp32 on the host for comparison.
            float *gpu_out = (float *)malloc(sizeof(float) * row_floats * N_ROWS);
            const uint16_t *h16 = (const uint16_t *)[bufOutF16 contents];
            for (size_t i = 0; i < row_floats * N_ROWS; i++) {
                // Use __fp16 if available; otherwise manual decode.
                __fp16 hh;
                memcpy(&hh, &h16[i], sizeof(hh));
                gpu_out[i] = (float)hh;
            }
            double cs_gpu_dec = cosine_similarity(input, gpu_out, row_floats * N_ROWS);
            fprintf(stdout, "bits=%d  GPU encode + GPU decode  cos_sim=%.6f\n", bits, cs_gpu_dec);

            // ----- GPU batch decode (single dispatch over all rows) -----
            float *gpu_batch = (float *)malloc(sizeof(float) * row_floats * N_ROWS);
            {
                id<MTLBuffer> bufOutBatch = [device newBufferWithLength:sizeof(uint16_t) * row_floats * N_ROWS
                                                              options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> dec = [cb computeCommandEncoder];
                [dec setComputePipelineState:ps_bat];
                [dec setBuffer:bufArgs   offset:0 atIndex:0];
                [dec setBuffer:bufEnc    offset:0 atIndex:1];
                [dec setBuffer:bufOutBatch offset:0 atIndex:2];
                [dec setThreadgroupMemoryLength:sizeof(float) * 128 atIndex:0];
                [dec setThreadgroupMemoryLength:sizeof(float) *   4 atIndex:1];
                MTLSize tg = MTLSizeMake(128, 1, 1);
                MTLSize grid = MTLSizeMake(N_ROWS, 1, 1);
                [dec dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                [dec endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.error) {
                    fprintf(stderr, "batch dispatch error: %s\n",
                            [[cb.error localizedDescription] UTF8String]);
                    overall_rc = 5;
                    free(cpu_out); free(cpu_enc); free(gpu_out); free(gpu_batch);
                    break;
                }
                const uint16_t *bb16 = (const uint16_t *)[bufOutBatch contents];
                for (size_t i = 0; i < row_floats * N_ROWS; i++) {
                    __fp16 hh;
                    memcpy(&hh, &bb16[i], sizeof(hh));
                    gpu_batch[i] = (float)hh;
                }
            }
            double cs_batch = cosine_similarity(input, gpu_batch, row_floats * N_ROWS);
            fprintf(stdout, "bits=%d  GPU encode + GPU batch    cos_sim=%.6f\n", bits, cs_batch);

            if (!(cs_gpu_enc > 0.95)) {
                fprintf(stderr, "FAIL bits=%d GPU encode round-trip (cos_sim=%.6f)\n",
                        bits, cs_gpu_enc);
                overall_rc = 6;
            }
            if (!(cs_gpu_dec > 0.95)) {
                fprintf(stderr, "FAIL bits=%d GPU encode+decode (cos_sim=%.6f)\n",
                        bits, cs_gpu_dec);
                overall_rc = 7;
            }
            if (!(cs_batch > 0.95)) {
                fprintf(stderr, "FAIL bits=%d GPU batch decode (cos_sim=%.6f)\n",
                        bits, cs_batch);
                overall_rc = 8;
            }

            free(cpu_out);
            free(cpu_enc);
            free(gpu_out);
            free(gpu_batch);
        }

        // ----- WHT-128 self-inverse smoke test -----
        if (overall_rc == 0) {
            float wht_in[128], wht_buf[128];
            for (int i = 0; i < 128; i++) wht_in[i] = (float)(i - 63) * 0.1f;
            memcpy(wht_buf, wht_in, sizeof(wht_in));

            id<MTLBuffer> bufData = [device newBufferWithBytes:wht_buf length:sizeof(wht_buf)
                                                      options:MTLResourceStorageModeShared];

            turbo_args_t a_fwd = (turbo_args_t){
                .head_dim = 128, .bits = 0, .n_rows = 1, .pad = 0,
                .row_bytes_in = 0, .row_bytes_out = 0,
            };
            id<MTLBuffer> bufA_fwd = [device newBufferWithBytes:&a_fwd length:sizeof(a_fwd)
                                                        options:MTLResourceStorageModeShared];
            {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps_wht];
                [e setBuffer:bufA_fwd offset:0 atIndex:0];
                [e setBuffer:bufData  offset:0 atIndex:1];
                [e setThreadgroupMemoryLength:sizeof(float) * 128 atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            turbo_args_t a_inv = a_fwd; a_inv.bits = 1;
            id<MTLBuffer> bufA_inv = [device newBufferWithBytes:&a_inv length:sizeof(a_inv)
                                                        options:MTLResourceStorageModeShared];
            {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps_wht];
                [e setBuffer:bufA_inv offset:0 atIndex:0];
                [e setBuffer:bufData  offset:0 atIndex:1];
                [e setThreadgroupMemoryLength:sizeof(float) * 128 atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            const float *wht_out = (const float *)[bufData contents];
            double max_err = 0.0;
            for (int i = 0; i < 128; i++) {
                double e = fabs((double)wht_in[i] - (double)wht_out[i]);
                if (e > max_err) max_err = e;
            }
            fprintf(stdout, "wht_inplace fwd+inv max abs err = %.6g\n", max_err);
            if (max_err > 1e-3) {
                fprintf(stderr, "FAIL WHT self-inverse error too large\n");
                overall_rc = 9;
            }
        }

        free(input);

        if (overall_rc == 0) fprintf(stdout, "OK\n");
        else                  fprintf(stderr, "FAIL rc=%d\n", overall_rc);
        return overall_rc;
    }
}
