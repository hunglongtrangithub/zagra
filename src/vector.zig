const std = @import("std");
const log = std.log.scoped(.vector);

const znpy = @import("znpy");

const types = @import("types.zig");

/// A generic N-dimensional vector type with elements of type T
pub fn Vector(comptime T: type, comptime N: usize) type {
    const elem_type = types.ElemType.fromZigType(T) orelse @compileError("Unsupported element type: " ++ @typeName(T));
    const dim_type = types.DimType.fromDim(N) orelse @compileError("Unsupported vector dimension");

    _ = dim_type;
    return struct {
        /// Aligned storage for the vector elements.
        /// 64 bytes alignment for SIMD performance.
        data: [N]T align(64),

        const Self = @This();

        /// Fill all elements of the vector with random values
        pub fn initRandom(rng: *std.Random) Self {
            var vec: Self = undefined;
            for (0..N) |i| {
                vec.data[i] = switch (elem_type) {
                    // .UInt8 => rng.int(T),
                    .Int8 => rng.int(T),
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

            var acc: Vec = @splat(0);

            // Handle chunks with SIMD
            inline for (0..num_chunks) |chunk_idx| {
                const i = chunk_idx * vector_size;
                const chunk1: Vec = v1.data[i..][0..vector_size].*;
                const chunk2: Vec = v2.data[i..][0..vector_size].*;
                const diff = chunk1 - chunk2;
                acc += diff * diff;
            }

            // Handle remainder elements
            var tail_acc: T = 0;
            if (remainder > 0) {
                inline for (0..remainder) |tail_idx| {
                    const i = num_chunks * vector_size + tail_idx;
                    const diff = v1.data[i] - v2.data[i];
                    tail_acc += diff * diff;
                }
            }

            const final_acc = @reduce(.Add, acc) + tail_acc;
            return final_acc;
        }
    };
}
