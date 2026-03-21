//! This is the root of the bench module. Benchmarks are in zig files whose names start with "bench_".
//! This file includes test for the csv module.
const csv = @import("csv.zig");

test {
    _ = csv;
}
