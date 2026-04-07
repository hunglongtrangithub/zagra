const std = @import("std");

/// A type for node identifiers, chosen based on the size of usize.
/// Safe to cast to usize.
pub const NodeIdType = switch (@sizeOf(usize)) {
    1 => u8,
    2 => u16,
    4 => u32,
    8 => u64,
    else => @compileError("Unsupported usize size"),
};

/// Element types supported for vector storage and distance computation.
///
/// Currently, Int8 and UInt8 are reserved for future support but commented out because:
/// - hnswlib (the reference HNSW implementation used as baseline) only
///   supports float32
/// - faiss and CAGRA support quantized types (i8/u8) for memory efficiency,
///   but this is a separate concern from the current benchmark focus
pub const ElemType = enum {
    /// 32-bit signed integer elements.
    Int32,
    /// 32-bit floating point elements (IEEE 754 binary32).
    Float,
    /// 16-bit floating point elements (IEEE 754 binary16, half precision).
    Half,
    // Int8,
    // UInt8,

    const Self = @This();

    /// Converts a Zig primitive type to its corresponding ElemType variant.
    /// Returns null if the type has no mapping (e.g., u8, f64).
    pub fn fromZigType(comptime T: type) ?Self {
        return switch (T) {
            i32 => Self.Int32,
            f32 => Self.Float,
            f16 => Self.Half,
            // i8 => Self.Int8,
            // u8 => Self.UInt8,
            else => null,
        };
    }

    /// Returns the Zig primitive type that corresponds to this element type.
    pub fn toZigType(self: Self) type {
        return switch (self) {
            .Int32 => i32,
            .Float => f32,
            .Half => f16,
            // .Int8 => i8,
            // .UInt8 => u8,
        };
    }
};

/// Vector dimensionality options for SIMD-optimized distance calculations.
///
/// Dimensions are chosen to be divisible by 64 to align with AVX-256
/// and AVX-512 vector registers, enabling efficient batch processing.
pub const DimType = enum {
    /// 128-dimensional vectors.
    D128,
    /// 256-dimensional vectors.
    D256,
    /// 512-dimensional vectors.
    D512,

    const Self = @This();

    /// Converts a dimension count to its corresponding DimType variant.
    /// Returns null if the dimension is not supported.
    pub fn fromDim(n: usize) ?Self {
        return switch (n) {
            128 => Self.D128,
            256 => Self.D256,
            512 => Self.D512,
            else => null,
        };
    }

    /// Returns the numeric dimension value for this variant.
    pub fn toDim(self: Self) usize {
        return switch (self) {
            .D128 => 128,
            .D256 => 256,
            .D512 => 512,
        };
    }
};

/// Computes the maximum absolute value for vector elements to ensure that the squared
/// distance between two vectors does not overflow.
///
/// This prevents arithmetic overflow when computing squared L2 distances:
/// `dist² = Σ(v1[i] - v2[i])²`.
///
/// The safe range is `[-max_val, max_val]` for all elements.
///
/// Arguments:
/// - `elem`: The element type of the vectors
/// - `dim`: The dimensionality of the vectors
pub fn maxAbsValue(comptime elem: ElemType, comptime dim: DimType) elem.toZigType() {
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
