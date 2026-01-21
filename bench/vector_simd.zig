const std = @import("std");

const zagra = @import("zagra");

/// Benchmark configuration
const BenchmarkConfig = struct {
    warmup_iterations: usize = 1000,
    measurement_iterations: usize = 1_000_000,
    name: []const u8,
};

/// Benchmark result statistics
const BenchmarkResult = struct {
    min_ns: u64,
    max_ns: u64,
    mean_ns: u64,
    median_ns: u64,
    stddev_ns: f64,

    pub fn print(self: BenchmarkResult, writer: anytype, name: []const u8) !void {
        try writer.print("{s}:\n", .{name});
        try writer.print("  Min:    {d:>10.2} ns\n", .{@as(f64, @floatFromInt(self.min_ns))});
        try writer.print("  Max:    {d:>10.2} ns\n", .{@as(f64, @floatFromInt(self.max_ns))});
        try writer.print("  Mean:   {d:>10.2} ns\n", .{@as(f64, @floatFromInt(self.mean_ns))});
        try writer.print("  Median: {d:>10.2} ns\n", .{@as(f64, @floatFromInt(self.median_ns))});
        try writer.print("  StdDev: {d:>10.2} ns\n", .{self.stddev_ns});
    }
};

/// Run a benchmark and collect statistics
fn runBenchmark(
    comptime T: type,
    comptime VecType: type,
    comptime func: fn (v1: *const VecType, v2: *const VecType) T,
    args: struct {
        v1: *const VecType,
        v2: *const VecType,
    },
    config: BenchmarkConfig,
    allocator: std.mem.Allocator,
) !BenchmarkResult {
    // Warmup
    for (0..config.warmup_iterations) |_| {
        const result = func(args.v1, args.v2);
        std.mem.doNotOptimizeAway(result);
    }

    // Allocate array for timing results
    var timings = try allocator.alloc(u64, config.measurement_iterations);
    defer allocator.free(timings);

    // Measurement
    var min: u64 = std.math.maxInt(u64);
    var max: u64 = 0;
    var total: u64 = 0;

    for (0..config.measurement_iterations) |i| {
        var timer = try std.time.Timer.start();
        const result = func(args.v1, args.v2);
        const elapsed = timer.read();
        std.mem.doNotOptimizeAway(result);

        timings[i] = elapsed;
        min = @min(min, elapsed);
        max = @max(max, elapsed);
        total += elapsed;
    }

    const mean = total / config.measurement_iterations;

    // Calculate standard deviation
    var variance_sum: f64 = 0;
    for (timings) |timing| {
        const diff = @as(f64, @floatFromInt(timing)) - @as(f64, @floatFromInt(mean));
        variance_sum += diff * diff;
    }
    const stddev = @sqrt(variance_sum / @as(f64, @floatFromInt(config.measurement_iterations)));

    // Calculate median
    std.mem.sort(u64, timings, {}, std.sort.asc(u64));
    const median = timings[config.measurement_iterations / 2];

    return BenchmarkResult{
        .min_ns = min,
        .max_ns = max,
        .mean_ns = mean,
        .median_ns = median,
        .stddev_ns = stddev,
    };
}

/// Run single benchmark variant
fn benchmarkVariant(
    comptime T: type,
    comptime VecType: type,
    comptime func: fn (v1: *const VecType, v2: *const VecType) T,
    name: []const u8,
    v1: *const VecType,
    v2: *const VecType,
    base_config: BenchmarkConfig,
    allocator: std.mem.Allocator,
    writer: *std.io.Writer,
) !BenchmarkResult {
    const config = BenchmarkConfig{
        .warmup_iterations = base_config.warmup_iterations,
        .measurement_iterations = base_config.measurement_iterations,
        .name = name,
    };

    const result = try runBenchmark(
        T,
        VecType,
        func,
        .{ .v1 = v1, .v2 = v2 },
        config,
        allocator,
    );

    try result.print(writer, name);
    try writer.print("\n", .{});

    return result;
}

/// Verify correctness of implementation against baseline
fn verifyCorrectness(
    comptime T: type,
    baseline: T,
    value: T,
    name: []const u8,
    writer: anytype,
) !bool {
    const tolerance = switch (@typeInfo(T)) {
        .float => std.math.floatEps(T) * 10,
        .int => 0,
        else => @compileError("Unsupported type: " ++ @typeName(T)),
    };

    const diff = if (@typeInfo(T) == .float)
        @abs(baseline - value)
    else
        @abs(@as(i64, baseline) - @as(i64, value));

    const ok = diff <= tolerance;

    if (ok) {
        try writer.print("   {s}: ✅ (diff: {d})\n", .{ name, diff });
    } else {
        try writer.print("   {s}: ❌ value={d}, diff={d}\n", .{ name, value, diff });
    }

    return ok;
}

/// Compare SIMD vs Naive implementation
fn benchmarkComparison(
    comptime T: type,
    comptime N: usize,
    allocator: std.mem.Allocator,
    writer: *std.io.Writer,
) !void {
    const VecType = zagra.Vector(T, N);

    // Initialize test vectors
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const v1 = VecType.initRandom(rng);
    const v2 = VecType.initRandom(rng);

    // Print header
    try writer.print("\n" ++ "=" ** 80 ++ "\n", .{});
    try writer.print("Benchmarking Vector({s}, {d})\n", .{ @typeName(T), N });
    try writer.print("=" ** 80 ++ "\n\n", .{});

    const base_config = BenchmarkConfig{
        .warmup_iterations = 1000_000,
        .measurement_iterations = 100_000_000,
        .name = "",
    };

    try writer.print("Config: warmup={d}, measurements={d}\n\n", .{
        base_config.warmup_iterations,
        base_config.measurement_iterations,
    });

    // Run benchmarks for all variants
    const naive_result = try benchmarkVariant(T, VecType, VecType.sqdistNaive, "Naive", &v1, &v2, base_config, allocator, writer);
    const simd1_result = try benchmarkVariant(T, VecType, VecType.sqdist, "SIMD-1 (1 register)", &v1, &v2, base_config, allocator, writer);
    const simd4_result = try benchmarkVariant(T, VecType, VecType.sqdistSIMD4, "SIMD-4 (4 registers)", &v1, &v2, base_config, allocator, writer);

    // Calculate and print speedups
    const speedup_simd1 = @as(f64, @floatFromInt(naive_result.mean_ns)) / @as(f64, @floatFromInt(simd1_result.mean_ns));
    const speedup_simd4 = @as(f64, @floatFromInt(naive_result.mean_ns)) / @as(f64, @floatFromInt(simd4_result.mean_ns));
    const simd4_vs_simd1 = @as(f64, @floatFromInt(simd1_result.mean_ns)) / @as(f64, @floatFromInt(simd4_result.mean_ns));

    try writer.print("Speedups:\n", .{});
    try writer.print("  SIMD-1 vs Naive:  {d:.2}x\n", .{speedup_simd1});
    try writer.print("  SIMD-4 vs Naive:  {d:.2}x\n", .{speedup_simd4});
    try writer.print("  SIMD-4 vs SIMD-1: {d:.2}x\n", .{simd4_vs_simd1});
    try writer.print("\n", .{});

    // Verify correctness
    const naive_val = VecType.sqdistNaive(&v1, &v2);
    const simd1_val = VecType.sqdist(&v1, &v2);
    const simd4_val = VecType.sqdistSIMD4(&v1, &v2);

    try writer.print("Correctness (baseline: {d}):\n", .{naive_val});
    const simd1_ok = try verifyCorrectness(T, naive_val, simd1_val, "SIMD-1", writer);
    const simd4_ok = try verifyCorrectness(T, naive_val, simd4_val, "SIMD-4", writer);

    if (simd1_ok and simd4_ok) {
        try writer.print("\n✅ All implementations verified correct\n", .{});
    } else {
        try writer.print("\n❌ Correctness check failed!\n", .{});
    }
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("\n", .{});
    try stdout.print("SIMD Vector Distance Benchmark Suite\n", .{});

    try benchmarkComparison(f32, 128, allocator, stdout);
    try stdout.flush();
    try benchmarkComparison(f32, 256, allocator, stdout);
    try stdout.flush();
    try benchmarkComparison(f32, 512, allocator, stdout);
    try stdout.flush();

    try stdout.print("\n" ++ "=" ** 80 ++ "\n", .{});
    try stdout.print("Benchmark complete!\n", .{});
    try stdout.print("=" ** 80 ++ "\n\n", .{});
    try stdout.flush();
}
