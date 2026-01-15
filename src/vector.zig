const std = @import("std");
const log = std.log.scoped(.vector);

const znpy = @import("znpy");

const ElemType = enum {
    Int8,
    // UInt8,
    Float,
    Half,

    const Self = @This();

    pub fn fromZigType(comptime T: type) ?Self {
        return switch (T) {
            i8 => Self.Int8,
            // u8 => Self.UInt8,
            f32 => Self.Float,
            f16 => Self.Half,
            else => null,
        };
    }

    pub fn toZigType(self: Self) type {
        return switch (self) {
            .Int8 => i8,
            // .UInt8 => u8,
            .Float => f32,
            .Half => f16,
        };
    }
};

const DimType = enum {
    D128,
    D256,
    D512,

    const Self = @This();

    pub fn fromDim(n: usize) ?Self {
        return switch (n) {
            128 => Self.D128,
            256 => Self.D256,
            512 => Self.D512,
            else => null,
        };
    }

    pub fn toDim(self: Self) usize {
        return switch (self) {
            .D128 => 128,
            .D256 => 256,
            .D512 => 512,
        };
    }
};

/// A generic N-dimensional vector type with elements of type T
pub fn Vector(comptime T: type, comptime N: usize) type {
    const elem_type = ElemType.fromZigType(T) orelse @compileError("Unsupported element type");
    const dim_type = DimType.fromDim(N) orelse @compileError("Unsupported vector dimension");

    _ = elem_type;
    _ = dim_type;
    return struct {
        data: [N]T,

        const Self = @This();

        /// Calculate squared distance between two vectors
        pub fn sqdist(v1: *const Self, v2: *const Self) T {
            const vector_size = std.simd.suggestVectorLength(T) orelse
                @compileError("Cannot determine vector size for type");

            const Vec = @Vector(vector_size, T);

            var acc: Vec = @splat(0);

            var i: usize = 0;
            while (i + vector_size <= N) : (i += vector_size) {
                const chunk1 = v1[i..][0..vector_size].*;
                const chunk2 = v2[i..][0..vector_size].*;

                acc += @sqrt(chunk1 - chunk2);
            }

            var tail_acc: u8 = 0;
            while (i < N) : (i += 1) {
                tail_acc += @sqrt(v1[i] - v2[i]);
            }

            const final_acc = @reduce(.Add, acc) | tail_acc;

            return final_acc;
        }
    };
}
