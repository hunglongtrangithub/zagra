//! root source file of the zagra module
const std = @import("std");

const vector = @import("vector.zig");
const dataset = @import("dataset.zig");

pub const Vector = vector.Vector;

test {
    _ = vector;
    _ = dataset;
}
