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
//
// The 3-bit format is a strict prefix of the 4-bit format; the qjl
// slot only carries the high bit of the 4-bit codebook index.

#define TURBO_ROT_D    128
#define TURBO_BLOCK    32

struct ds4_metal_args_turbo {
    int32_t  head_dim;
    int32_t  bits;          // 3 or 4 (for wht_inplace: 0 = forward, 1 = inverse)
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
    if (bits != 3 && bits != 4) return;

    const int n_rot     = head_dim / TURBO_ROT_D;
    const int blocks_pg = TURBO_ROT_D / TURBO_BLOCK; // 4
    const uint blk_bytes = (bits == 3) ? 14u : 18u;

    device const float * src_row = src + (uint)row * (uint)head_dim;
    device       uchar * dst_row = dst + (uint)row * (uint)blk_bytes * (uint)(n_rot * blocks_pg);

    const ushort b    = tid >> 5;    // 0..3 (block index)
    const ushort lane = tid & 31;    // 0..31 (lane within block)

    threadgroup float * sq_buf      = scratch + 4;
    threadgroup uint  * stage       = (threadgroup uint *)(scratch + 136);
    threadgroup uint  * pack        = (threadgroup uint *)(scratch + 656);

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
        const int idx = (bits == 3) ? turbo_nearest3(val) : turbo_nearest4(val);

        // Stage per-lane bit contributions.
        //   qs   = uint64 packed as two uint32 (qs_lo, qs_hi):
        //          lane i (i < 16) contributes (idx & 3) at bit ((i & 15)*2) of qs_lo;
        //          lane i (i >= 16) contributes (idx & 3) at bit ((i & 15)*2) of qs_hi.
        //   signs= uint32: lane i contributes bit 2 of idx at bit position i.
        //   qjl  = uint32 (4-bit only): lane i contributes bit 3 of idx at bit position i.
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

        // One writer per block emits the encoded bytes. Norm is held
        // in `norm` (broadcasted via sq_buf above, so every lane has
        // it consistently).
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
    const uint blk_bytes = (bits == 3) ? 14u : 18u;

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

    const uchar qs_byte = blk[2 + (lane >> 2)];
    const uchar sg_byte = blk[10 + (lane >> 3)];
    const uchar lo      = (qs_byte >> ((lane & 3) * 2)) & 0x3;
    const uchar hi      = (sg_byte >> (lane & 7)) & 0x1;
    int idx;
    if (bits == 3) {
        idx = (int)(lo | (hi << 2));
    } else {
        const uchar qj_byte = blk[14 + (lane >> 3)];
        const uchar hi2 = (qj_byte >> (lane & 7)) & 0x1;
        idx = (int)(lo | (hi << 2) | (hi2 << 3));
    }

    const float c = (bits == 3) ? turbo_cb3[idx] : turbo_cb4[idx];
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
    if (bits != 3 && bits != 4) return;

    const int n_rot     = head_dim / TURBO_ROT_D;
    const int blocks_pg = TURBO_ROT_D / TURBO_BLOCK;
    const uint blk_bytes = (bits == 3) ? 14u : 18u;
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
    if (bits != 3 && bits != 4) return;

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
