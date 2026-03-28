//! This is the root of the bench module. Benchmarks are in zig files whose names start with "bench_".
//! This file includes test for the csv module.
const config = @import("config");
const csv = @import("csv.zig");

/// Directory to store result files from running benchmarks
pub const RESULTS_DIR: []const u8 = config.BENCH_DIR ++ "/results";

test {
    _ = csv;
}
