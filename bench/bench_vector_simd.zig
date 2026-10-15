const std = @import("std");

const zagra = @import("zagra");
const csv = @import("csv.zig");
const help = @import("help.zig");
const root = @import("root.zig");

/// Section separator used in printed output.
const separator: [80]u8 = @splat('=');

/// Benchmark configuration
const BenchmarkConfig = struct {
    warmup_iterations: usize = 1000_000,
    measurement_iterations: usize = 100_000_000,
    name: []const u8,
};

/// Benchmark result statistics
const BenchmarkResult = struct {
    min_ns: std.Io.Duration,
    max_ns: std.Io.Duration,
    mean_ns: std.Io.Duration,
    median_ns: std.Io.Duration,
    stddev_ns: f64,

    pub fn print(self: BenchmarkResult, writer: *std.Io.Writer, name: []const u8) !void {
        try writer.print("{s}:\n", .{name});
        try writer.print("  Min:    {d:>10.2} ns\n", .{@as(f64, @floatFromInt(self.min_ns.toNanoseconds()))});
        try writer.print("  Max:    {d:>10.2} ns\n", .{@as(f64, @floatFromInt(self.max_ns.toNanoseconds()))});
        try writer.print("  Mean:   {d:>10.2} ns\n", .{@as(f64, @floatFromInt(self.mean_ns.toNanoseconds()))});
        try writer.print("  Median: {d:>10.2} ns\n", .{@as(f64, @floatFromInt(self.median_ns.toNanoseconds()))});
        try writer.print("  StdDev: {d:>10.2} ns\n", .{self.stddev_ns});
    }
};

const ComparisonResult = struct {
    element_type: []const u8,
    vector_length: usize,
    naive: BenchmarkResult,
    simd: BenchmarkResult,
    speedup: f64,
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
    io: std.Io,
) !BenchmarkResult {
    // Warmup
    for (0..config.warmup_iterations) |_| {
        const result = func(args.v1, args.v2);
        std.mem.doNotOptimizeAway(result);
    }

    // Allocate array for timing results
    var timings = try allocator.alloc(std.Io.Duration, config.measurement_iterations);
    defer allocator.free(timings);

    // Measurement
    var min: std.Io.Duration = .max;
    var max: std.Io.Duration = .zero;
    var total: i96 = 0;

    var prev = std.Io.Timestamp.now(io, .awake);
    for (0..config.measurement_iterations) |i| {
        const result = func(args.v1, args.v2);
        const now = std.Io.Timestamp.now(io, .awake);
        std.mem.doNotOptimizeAway(result);

        const elapsed = prev.durationTo(now);
        prev = now;

        timings[i] = elapsed;
        if (elapsed.nanoseconds < min.nanoseconds) min = elapsed;
        if (elapsed.nanoseconds > max.nanoseconds) max = elapsed;
        total += elapsed.nanoseconds;
    }

    const mean: std.Io.Duration = .fromNanoseconds(
        @divTrunc(total, @as(i96, @intCast(config.measurement_iterations))),
    );

    // Calculate standard deviation
    var variance_sum: f64 = 0;
    for (timings) |timing| {
        const diff = @as(f64, @floatFromInt(timing.nanoseconds)) - @as(f64, @floatFromInt(mean.nanoseconds));
        variance_sum += diff * diff;
    }
    const stddev = @sqrt(variance_sum / @as(f64, @floatFromInt(config.measurement_iterations)));

    // Calculate median
    std.mem.sort(std.Io.Duration, timings, {}, struct {
        fn lt(_: void, lhs: std.Io.Duration, rhs: std.Io.Duration) bool {
            return lhs.nanoseconds < rhs.nanoseconds;
        }
    }.lt);
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
    writer: *std.Io.Writer,
    io: std.Io,
) !ComparisonResult {
    const VecType = zagra.Vector(T, N);

    // Initialize test vectors
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const v1 = VecType.initRandom(random);
    const v2 = VecType.initRandom(random);

    try writer.print("\n{s}\n", .{separator});
    try writer.print("Benchmarking Vector({}, {d})\n", .{ T, N });
    try writer.print("\n{s}\n", .{separator});
    try writer.flush();

    // Benchmark naive implementation
    const naive_config = BenchmarkConfig{
        .name = "Naive Implementation",
    };
    try writer.print("Bench Config: warmup_iterations={d}, measurement_iterations={d}\n", .{ naive_config.warmup_iterations, naive_config.measurement_iterations });
    try writer.flush();
    const naive_result = try runBenchmark(
        T,
        VecType,
        VecType.sqdistNaive,
        .{ .v1 = &v1, .v2 = &v2 },
        naive_config,
        allocator,
        io,
    );
    try naive_result.print(writer, "Naive Implementation");
    try writer.flush();

    try writer.print("\n", .{});

    // Benchmark SIMD implementation
    try writer.print("Bench Config: warmup_iterations={d}, measurement_iterations={d}\n", .{ naive_config.warmup_iterations, naive_config.measurement_iterations });
    try writer.flush();
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
        io,
    );
    try simd_result.print(writer, "SIMD Implementation");
    try writer.flush();

    // Calculate speedup
    const speedup = @as(f64, @floatFromInt(naive_result.mean_ns.toNanoseconds())) /
        @as(f64, @floatFromInt(simd_result.mean_ns.toNanoseconds()));

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

    const type_name = switch (@typeInfo(T)) {
        .float => "f32",
        .int => @typeName(T),
        else => @compileError("Unsupported type"),
    };

    return ComparisonResult{
        .element_type = type_name,
        .vector_length = N,
        .naive = naive_result,
        .simd = simd_result,
        .speedup = 1.0,
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const result_name = "vector_simd";

    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();
    const exe_path = args.next() orelse @src().file;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    const result_id = args.next();
    try help.checkHelp(stdout, result_id, exe_path);

    try stdout.print("\n", .{});
    try stdout.print("SIMD Vector Distance Benchmark Suite\n", .{});

    const results = [_]ComparisonResult{
        try benchmarkComparison(f32, 128, allocator, stdout, io),
        try benchmarkComparison(f32, 256, allocator, stdout, io),
        try benchmarkComparison(f32, 512, allocator, stdout, io),
    };

    try stdout.print("\n{s}\n", .{separator});
    try stdout.print("Benchmark complete!\n", .{});
    try stdout.print("\n{s}\n", .{separator});

    try stdout.flush();

    const results_dir = root.RESULTS_DIR;
    try std.Io.Dir.cwd().createDirPath(io, results_dir);

    const csv_file_name = if (result_id) |id|
        try allocator.print("{s}/{s}_{s}.csv", .{ results_dir, result_name, id })
    else
        try allocator.print("{s}/{s}_{d}.csv", .{ results_dir, result_name, std.Io.Timestamp.now(io, .real).nanoseconds });
    const csv_file = try std.Io.Dir.cwd().createFile(io, csv_file_name, .{});
    defer csv_file.close(io);

    var file_buffer: [1024]u8 = undefined;
    var file_writer = csv_file.writer(io, &file_buffer);
    const file = &file_writer.interface;

    const headers = &[_][]const u8{
        "vector_type",
        "element_type",
        "vector_length",
        "min_ns",
        "max_ns",
        "mean_ns",
        "median_ns",
        "stddev_ns",
        "speedup",
    };
    try csv.writeHeaders(file, headers);
    for (results) |result| {
        try csv.writeRow(
            file,
            .{
                "naive",
                result.element_type,
                result.vector_length,
                result.naive.min_ns.toNanoseconds(),
                result.naive.max_ns.toNanoseconds(),
                result.naive.mean_ns.toNanoseconds(),
                result.naive.median_ns.toNanoseconds(),
                result.naive.stddev_ns,
                result.speedup,
            },
            headers.len,
        );
        try csv.writeRow(
            file,
            .{
                "simd",
                result.element_type,
                result.vector_length,
                result.simd.min_ns.toNanoseconds(),
                result.simd.max_ns.toNanoseconds(),
                result.simd.mean_ns.toNanoseconds(),
                result.simd.median_ns.toNanoseconds(),
                result.simd.stddev_ns,
                result.speedup,
            },
            headers.len,
        );
    }
    try file.flush();

    try stdout.print("Results written to {s}\n", .{csv_file_name});
    try stdout.flush();
}
