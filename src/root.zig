//! root source file of the zagra module
const std = @import("std");

const DataType = enum {
    /// Zig's f32 type
    f32,
    /// Zig's i32 type
    i32,
};

fn getType(comptime dt: DataType) type {
    return switch (dt) {
        .f32 => f32,
        .i32 => i32,
    };
}

const Dimension = enum(usize) {
    /// 3-dimensional vector
    D3 = 3,
    /// 32-dimensional vector
    D32 = 32,
    /// 64-dimensional vector
    D64 = 64,
    /// 128-dimensional vector
    D128 = 128,
};

fn getDim(comptime dim: Dimension) usize {
    return switch (dim) {
        .D3 => 3,
        .D32 => 32,
        .D64 => 64,
        .D128 => 128,
    };
}

/// A generic N-dimensional vector type with elements of type T
pub fn Vector(comptime data_type: DataType, comptime dimension: Dimension) type {
    const T = getType(data_type);
    const N = getDim(dimension);
    return struct {
        data: [N]T,

        const Self = @This();

        /// Initialize the vector with an array of values
        /// `N` must match the dimension of the vector.
        /// `T` must match the data type of the vector.
        pub fn init(values: [N]T) Self {
            return Self{ .data = values };
        }

        /// Access element at index `idx`
        pub fn at(self: *const Self, idx: usize) T {
            std.debug.assert(idx < N);
            return self.data[idx];
        }

        /// Calculate squared distance between two vectors
        pub fn sqdist(v1: *const Self, v2: *const Self) T {
            var sum: T = 0;
            // TODO: Optimize with SIMD later
            for (v1.data, v2.data) |e1, e2| {
                const diff = e1 - e2;
                sum += diff * diff;
            }
            return sum;
        }
    };
}
