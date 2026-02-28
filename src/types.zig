const std = @import("std");

pub const ElemType = enum {
    // Int8,
    // UInt8,
    Int32,
    Float,
    Half,

    const Self = @This();

    pub fn fromZigType(comptime T: type) ?Self {
        return switch (T) {
            // i8 => Self.Int8,
            // u8 => Self.UInt8,
            i32 => Self.Int32,
            f32 => Self.Float,
            f16 => Self.Half,
            else => null,
        };
    }

    pub fn toZigType(self: Self) type {
        return switch (self) {
            // .Int8 => i8,
            // .UInt8 => u8,
            .Int32 => i32,
            .Float => f32,
            .Half => f16,
        };
    }
};

/// A enum of supported vector lengths (number of elements in a vector).
/// Should be divisible by 64 for efficient SIMD vector distance calculation.
pub const DimType = enum {
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

/// Computes the maximum absolute value for vector elements to ensure that the squared distance between two vectors does not overflow.
/// Formula: max_val = sqrt(T::MAX) / sqrt(N * 4) -> range: [-max_val, max_val]
pub fn maxAbsValue(comptime elem: ElemType, comptime dim: DimType) switch (elem) {
    .Int32 => i32,
    .Float => f32,
    .Half => f16,
} {
    const N = dim.toDim();
    return switch (elem) {
        .Int32 => blk: {
            const max_int = std.math.maxInt(i32);
            const denom = @as(f64, @floatFromInt(N)) * 4.0;
            const val_f64 = @sqrt(@as(f64, @floatFromInt(max_int)) / denom);
            break :blk @as(i32, @intFromFloat(@floor(val_f64)));
        },
        .Float => blk: {
            const max_float = std.math.floatMax(f32);
            const denom = @as(f32, @floatFromInt(N)) * 4.0;
            break :blk @sqrt(max_float / denom);
        },
        .Half => blk: {
            const max_float = std.math.floatMax(f16);
            const denom = @as(f16, @floatFromInt(N)) * 4.0;
            break :blk @sqrt(max_float / denom);
        },
    };
}
