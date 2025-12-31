//! root source file of the zagra module
const std = @import("std");

const DataType = enum { f32, i32 };
const Dimension = enum(u32) { D2 = 2, D3 = 3, D4 = 4 };

fn getType(comptime dt: DataType) type {
    return switch (dt) {
        .f32 => f32,
        .i32 => i32,
    };
}

fn getDim(comptime dim: Dimension) u32 {
    return @intFromEnum(dim);
}
pub fn Vector(comptime T: type, comptime N: u32) type {
    return struct {
        data: [N]T,

        const Self = @This();

        pub fn init(values: [N]T) Self {
            return Self{ .data = values };
        }

        pub fn at(self: *const Self, idx: usize) T {
            return self.data[idx];
        }

        pub fn sqdist(a: *const Self, b: *const Self) T {
            var sum: T = 0;
            for (a.data, b.data) |x, y| {
                const diff = x - y;
                sum += diff * diff;
            }
            return sum;
        }
    };
}
