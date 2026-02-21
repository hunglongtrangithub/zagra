const std = @import("std");

const zagra = @import("zagra");

/// Benchmark configuration
const BenchmarkConfig = struct {
    warmup_iterations: usize = 1000_000,
    measurement_iterations: usize = 100_000_000,
    name: []const u8,
};

/// Benchmark result statistics
const BenchmarkResult = struct {
    min_ns: u64,
    max_ns: u64,
    mean_ns: u64,
    median_ns: u64,
    stddev_ns: f64,

    pub fn print(self: BenchmarkResult, writer: *std.io.Writer, name: []const u8) !void {
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
    var random = prng.random();

    const v1 = VecType.initRandom(&random);
    const v2 = VecType.initRandom(&random);

    try writer.print("\n" ++ "=" ** 80 ++ "\n", .{});
    try writer.print("Benchmarking Vector({}, {d})\n", .{ T, N });
    try writer.print("\n" ++ "=" ** 80 ++ "\n", .{});

    // Benchmark naive implementation
    const naive_config = BenchmarkConfig{
        .name = "Naive Implementation",
    };
    try writer.print("Bench Config: warmup_iterations={d}, measurement_iterations={d}\n", .{ naive_config.warmup_iterations, naive_config.measurement_iterations });
    const naive_result = try runBenchmark(
        T,
        VecType,
        VecType.sqdistNaive,
        .{ .v1 = &v1, .v2 = &v2 },
        naive_config,
        allocator,
    );
    try naive_result.print(writer, "Naive Implementation");

    try writer.print("\n", .{});

    // Benchmark SIMD implementation
    try writer.print("Bench Config: warmup_iterations={d}, measurement_iterations={d}\n", .{ naive_config.warmup_iterations, naive_config.measurement_iterations });
    const simd_config = BenchmarkConfig{
        .name = "SIMD Implementation",
    };
    const simd_result = try runBenchmark(
        T,
        VecType,
        VecType.sqdist,
        .{ .v1 = &v1, .v2 = &v2 },
        simd_config,
        allocator,
    );
    try simd_result.print(writer, "SIMD Implementation");

    // Calculate speedup
    const speedup = @as(f64, @floatFromInt(naive_result.mean_ns)) /
        @as(f64, @floatFromInt(simd_result.mean_ns));

    try writer.print("\n", .{});
    try writer.print("Speedup: {d:.2}x\n", .{speedup});

    // Verify correctness
    const naive_val = VecType.sqdistNaive(&v1, &v2);
    const simd_val = VecType.sqdist(&v1, &v2);

    const tolerance = switch (@typeInfo(T)) {
        .float => std.math.floatEps(T),
        .int => 0,
        else => @compileError("Unsupported type"),
    };

    const diff = if (@typeInfo(T) == .float)
        @abs(naive_val - simd_val)
    else
        @abs(@as(i64, naive_val) - @as(i64, simd_val));

    if (diff <= tolerance) {
        try writer.print("✅ Correctness verified (diff: {d})\n", .{diff});
    } else {
        try writer.print("❌ MISMATCH: naive={d}, simd={d}, diff={d}\n", .{ naive_val, simd_val, diff });
    }

    try writer.flush();
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("\n", .{});
    try stdout.print("SIMD Vector Distance Benchmark Suite\n", .{});

    try benchmarkComparison(f32, 128, allocator, stdout);
    try benchmarkComparison(f32, 256, allocator, stdout);
    try benchmarkComparison(f32, 512, allocator, stdout);

    try stdout.print("\n" ++ "=" ** 80 ++ "\n", .{});
    try stdout.print("Benchmark complete!\n", .{});
    try stdout.print("\n" ++ "=" ** 80 ++ "\n", .{});

    try stdout.flush();
}
