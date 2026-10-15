const std = @import("std");
const znpy = @import("znpy");
const zagra = @import("zagra");
const csv = @import("csv.zig");
const help = @import("help.zig");
const root = @import("root.zig");

pub const std_options: std.Options = .{
    .log_level = .err,
};

const BenchmarkConfig = struct {
    vector_count: usize = 1_000_000,
    graph_degrees: []const usize = &.{ 4, 8, 16, 32, 64, 128, 256 },
    block_size: usize = 16384,
    seed: u64 = 42,
    num_threads: usize,
};

const BenchmarkResult = struct {
    vector_count: usize,
    graph_degree: usize,
    intermediate_degree: usize,
    block_size: usize,
    num_threads: usize,
    timing: zagra.index.BuildTiming,
};

fn runBenchmark(
    comptime T: type,
    comptime N: usize,
    dataset: zagra.Dataset(T, N),
    num_threads: usize,
    seed: u64,
    block_size: usize,
    graph_degree: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
) !BenchmarkResult {
    const intermediate_degree = graph_degree * 2;

    const build_config = zagra.index.BuildConfig.init(
        graph_degree,
        intermediate_degree,
        dataset.len,
        num_threads,
        seed,
        block_size,
    );

    const idx, const timing = try zagra.Index(T, N).buildWithTiming(
        dataset,
        build_config,
        allocator,
        io,
    );
    defer allocator.free(idx.graph);

    return BenchmarkResult{
        .vector_count = dataset.len,
        .graph_degree = graph_degree,
        .intermediate_degree = intermediate_degree,
        .block_size = block_size,
        .num_threads = num_threads,
        .timing = timing,
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();
    const exe_path = args.next() orelse @src().file;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    const result_id = args.next();
    try help.checkHelp(stdout, result_id, exe_path);

    const T: type = f32;
    const N: usize = 128;

    const config = BenchmarkConfig{ .num_threads = std.Thread.getCpuCount() catch 1 };

    try stdout.print("Allocating dataset of {} vectors...\n", .{config.vector_count});
    try stdout.flush();

    var prng = std.Random.DefaultPrng.init(config.seed);
    const dataset = try zagra.Dataset(T, N).initRandom(
        config.vector_count,
        prng.random(),
        allocator,
    );
    defer dataset.deinit(allocator);

    try stdout.print("Running benchmarks...\n\n", .{});
    try stdout.flush();

    var all_results = std.ArrayList(BenchmarkResult).empty;
    defer all_results.deinit(allocator);

    for (config.graph_degrees) |graph_degree| {
        try stdout.print("Benchmarking: {} vectors, graph_degree {}...\n", .{ config.vector_count, graph_degree });
        try stdout.flush();

        const result = try runBenchmark(
            T,
            N,
            dataset,
            config.num_threads,
            config.seed,
            config.block_size,
            graph_degree,
            allocator,
            io,
        );
        try all_results.append(allocator, result);
    }

    const results_dir = root.RESULTS_DIR;
    std.Io.Dir.cwd().createDirPath(io, results_dir) catch |e| switch (e) {
        error.PathAlreadyExists => {},
        else => return e,
    };
    const result_name = "zagra_index_build";
    const csv_file_name = if (result_id) |id|
        try allocator.print("{s}/{s}_{s}.csv", .{ results_dir, result_name, id })
    else
        try allocator.print("{s}/{s}_{d}.csv", .{ results_dir, result_name, std.Io.Timestamp.now(io, .real).nanoseconds });
    defer allocator.free(csv_file_name);
    const csv_file = try std.Io.Dir.cwd().createFile(io, csv_file_name, .{});
    defer csv_file.close(io);
    var csv_buffer: [1024]u8 = undefined;
    var csv_writer = csv_file.writer(io, &csv_buffer);
    const csv_out = &csv_writer.interface;

    const headers = &[_][]const u8{
        "vector_count",
        "graph_degree",
        "intermediate_degree",
        "block_size",
        "num_threads",
        "nn_descent_s",
        "resource_free_s",
        "optimizer_s",
        "total_s",
    };
    try csv.writeHeaders(csv_out, headers);

    for (all_results.items) |result| {
        try csv.writeRow(
            csv_out,
            .{
                result.vector_count,
                result.graph_degree,
                result.intermediate_degree,
                result.block_size,
                result.num_threads,
                @as(f64, @floatFromInt(result.timing.nn_descent_ns)) / std.time.ns_per_s,
                @as(f64, @floatFromInt(result.timing.resource_free_ns)) / std.time.ns_per_s,
                @as(f64, @floatFromInt(result.timing.optimizer_ns)) / std.time.ns_per_s,
                @as(f64, @floatFromInt(result.timing.total_ns)) / std.time.ns_per_s,
            },
            headers.len,
        );
    }
    try csv_out.flush();

    try stdout.print("\nBenchmarks completed.\nResults written to {s}\n", .{csv_file_name});
    try stdout.flush();
}
