//! root source file of the zagra module
const std = @import("std");

pub const vector = @import("vector.zig");
pub const dataset = @import("dataset.zig");
pub const types = @import("types.zig");

pub const Vector = vector.Vector;

test {
    _ = vector;
    _ = dataset;
    _ = types;
}
