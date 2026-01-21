const std = @import("std");
const log = std.log.scoped(.vector);

const znpy = @import("znpy");

const types = @import("types.zig");

/// A generic N-dimensional vector type with elements of type T
pub fn Vector(comptime T: type, comptime N: usize) type {
    const elem_type = types.ElemType.fromZigType(T) orelse @compileError("Unsupported element type: " ++ @typeName(T));
    const dim_type = types.DimType.fromDim(N) orelse
        @compileError(std.fmt.comptimePrint("Unsupported vector dimension: {d}", .{N}));

    _ = dim_type;
    return struct {
        /// Aligned storage for the vector elements.
        /// 64 bytes alignment for SIMD performance.
        // Alignment of 64 bytes satisfies all natural alignments of types we support.
        data: [N]T align(64),

        const Self = @This();

        /// Fill all elements of the vector with random values
        pub fn initRandom(rng: std.Random) Self {
            var vec: Self = undefined;
            for (0..N) |i| {
                vec.data[i] = switch (elem_type) {
                    .Int32 => rng.intRangeAtMost(T, -100, 100),
                    .Float => rng.float(T) * 100,
                    .Half => rng.float(T) * 100,
                };
            }
            return vec;
        }

        /// Naive implementation (no SIMD)
        pub fn sqdistNaive(v1: *const Self, v2: *const Self) T {
            var acc: T = 0;
            for (v1.data, v2.data) |a, b| {
                const diff = a - b;
                acc += diff * diff;
            }
            return acc;
        }

        /// Calculate squared distance between two vectors with SIMD optimization
        pub fn sqdist(v1: *const Self, v2: *const Self) T {
            // TODO: Handle potential overflow for high-dimensional or large-magnitude data.
            // Consider using a wider accumulator (f64 or u64) or pre-normalizing vectors.
            const vector_size = std.simd.suggestVectorLength(T) orelse
                // TODO: Fallback to some default vector size?
                @compileError("Cannot determine vector size for type");
            const Vec = @Vector(vector_size, T);

            const num_chunks = N / vector_size;
            const remainder = N % vector_size;

            const v1_ptr: [*]align(64) const T = &v1.data;
            const v2_ptr: [*]align(64) const T = &v2.data;

            var acc: Vec = @splat(0);

            // Handle chunks with SIMD
            inline for (0..num_chunks) |chunk_idx| {
                const i = chunk_idx * vector_size;
                const chunk1: Vec = v1_ptr[i..][0..vector_size].*;
                const chunk2: Vec = v2_ptr[i..][0..vector_size].*;
                const diff = chunk1 - chunk2;
                acc += diff * diff;
            }

            // Handle remainder elements
            var tail_acc: T = 0;
            if (remainder > 0) {
                const idx_start = num_chunks * vector_size;
                inline for (0..remainder) |idx_tail| {
                    const i = idx_start + idx_tail;
                    const diff = v1_ptr[i] - v2_ptr[i];
                    tail_acc += diff * diff;
                }
            }

            const final_acc = @reduce(.Add, acc) + tail_acc;
            return final_acc;
        }

        /// Calculates the Squared Euclidean Distance using SIMD with 4-way parallel accumulators.
        /// Designed for Mechanical Sympathy: bypasses instruction latency and saturates ALU ports.
        pub fn sqdistSIMD4(v1: *const Self, v2: *const Self) T {
            const vector_size = std.simd.suggestVectorLength(T) orelse
                // TODO: Fallback to some default vector size?
                @compileError("Cannot determine vector size for type");

            const v1_ptr: [*]align(64) const T = &v1.data;
            const v2_ptr: [*]align(64) const T = &v2.data;

            const Vec = @Vector(vector_size, T);

            // Multiple accumulators to break the "Read-After-Write" dependency chain.
            // This allows the CPU to calculate 4 vector additions in parallel.
            var acc0: Vec = @splat(0);
            var acc1: Vec = @splat(0);
            var acc2: Vec = @splat(0);
            var acc3: Vec = @splat(0);

            const stride = vector_size * 4;
            var i: usize = 0;

            // 1. Primary Loop: Process 4 vectors at a time
            while (i + stride <= N) : (i += stride) {
                // Load and subtract 4 chunks.
                // Modern CPUs will fire these instructions almost simultaneously.
                // Wrap the array dereference in Vec() to tell Zig to use SIMD registers
                const d0: Vec = @as(Vec, v1_ptr[i + vector_size * 0 ..][0..vector_size].*) - @as(Vec, v2_ptr[i + vector_size * 0 ..][0..vector_size].*);
                const d1: Vec = @as(Vec, v1_ptr[i + vector_size * 1 ..][0..vector_size].*) - @as(Vec, v2_ptr[i + vector_size * 1 ..][0..vector_size].*);
                const d2: Vec = @as(Vec, v1_ptr[i + vector_size * 2 ..][0..vector_size].*) - @as(Vec, v2_ptr[i + vector_size * 2 ..][0..vector_size].*);
                const d3: Vec = @as(Vec, v1_ptr[i + vector_size * 3 ..][0..vector_size].*) - @as(Vec, v2_ptr[i + vector_size * 3 ..][0..vector_size].*);

                acc0 += d0 * d0;
                acc1 += d1 * d1;
                acc2 += d2 * d2;
                acc3 += d3 * d3;
            }

            // 2. Middle Path: Handle remaining full vectors (1-3 chunks left)
            while (i + vector_size <= N) : (i += vector_size) {
                const d = @as(Vec, v1_ptr[i..][0..vector_size].*) - @as(Vec, v2_ptr[i..][0..vector_size].*);
                acc0 += d * d;
            }

            // 3. Tail Path: Handle scalar remainder (if N is not a multiple of vector_size)
            var tail_acc: T = 0;
            while (i < N) : (i += 1) {
                const d = v1_ptr[i] - v2_ptr[i];
                tail_acc += d * d;
            }

            // 4. Reduction: Sum the vectors into a single scalar
            // We sum the accumulators first to keep the logic in the vector registers
            // as long as possible before the final horizontal reduction.
            const final_vec = (acc0 + acc1) + (acc2 + acc3);
            return @reduce(.Add, final_vec) + tail_acc;
        }
    };
}

test "Vector Squared Distance Correctness" {
    const Rng = std.Random.DefaultPrng;
    var prng = Rng.init(42);
    const random = prng.random();

    const dims = [_]usize{ 128, 256, 512 };

    inline for (dims) |N| {
        const V = Vector(i32, N);

        const v1 = V.initRandom(random);
        const v2 = V.initRandom(random);

        // 1. Ground Truth
        const expected = v1.sqdistNaive(&v2);

        // 2. Standard SIMD
        const actual_simd = v1.sqdist(&v2);

        // 3. 4-way Parallel Accumulator SIMD
        const actual_simd4 = v1.sqdistSIMD4(&v2);

        try std.testing.expectEqual(expected, actual_simd);
        try std.testing.expectEqual(expected, actual_simd4);
    }
}

test "Vector Squared Distance: Edge Cases" {
    const N = 256;
    const V = Vector(i32, N);

    // Test exact zeros
    var v_zero: V = undefined;
    @as(*[N]i32, &v_zero.data).* = [_]i32{0} ** N;

    // Test specific values to verify (a-b)^2 logic
    var v_ones: V = undefined;
    @as(*[N]i32, &v_ones.data).* = [_]i32{1} ** N;

    var v_neg_ones: V = undefined;
    @as(*[N]i32, &v_neg_ones.data).* = [_]i32{-1} ** N;

    // (1 - (-1))^2 = 2^2 = 4. 4 * 256 elements = 1024
    try std.testing.expectEqual(0, v_zero.sqdistSIMD4(&v_zero));
    try std.testing.expectEqual(1024, v_ones.sqdistSIMD4(&v_neg_ones));
}
