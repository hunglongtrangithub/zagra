const std = @import("std");
const znpy = @import("znpy");
const zagra = @import("zagra");
const csv = @import("csv.zig");
const help = @import("help.zig");

pub const std_options: std.Options = .{
    .log_level = .err,
};

const BenchmarkConfig = struct {
    vector_counts: []const usize,
    graph_degrees: []const usize,
    num_threads: usize = 4,
    max_iterations: usize = 30,
    delta: f32 = 0.001,
    seed: u64 = 42,
};

const BenchmarkResult = struct {
    vector_count: usize,
    graph_degree: usize,
    timing: zagra.index.NNDTrainingTiming,

    fn deinit(self: *BenchmarkResult, allocator: std.mem.Allocator) void {
        self.timing.deinit(allocator);
    }
};

fn runBenchmark(
    comptime T: type,
    comptime N: usize,
    dataset: zagra.dataset.Dataset(T, N),
    graph_degree: usize,
    config: BenchmarkConfig,
    allocator: std.mem.Allocator,
) !BenchmarkResult {
    const NNDescent = zagra.index.NNDescent(T, N);
    var training_config = zagra.index.NNDTrainingConfig.init(
        config.num_threads,
        dataset.len,
        null,
        config.seed,
    );
    training_config.num_neighbors_per_node = graph_degree;
    training_config.max_iterations = config.max_iterations;
    training_config.delta = config.delta;

    var nn_descent = try NNDescent.init(&dataset, training_config, allocator);
    defer nn_descent.deinit(allocator);

    const timing = try nn_descent.trainWithTiming(allocator);

    return BenchmarkResult{
        .vector_count = dataset.len,
        .graph_degree = graph_degree,
        .timing = timing,
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    const exe_path = args.next() orelse @src().file;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    const result_prefix = args.next();
    try help.checkHelp(stdout, result_prefix, exe_path);

    const T: type = f32; // Element type
    const N: usize = 128; // Vector length

    // Benchmark configuration
    const bench_config = BenchmarkConfig{
        .vector_counts = &[_]usize{ 1_000_000, 5_000_000 },
        .graph_degrees = &[_]usize{ 8, 16, 32, 64 },
        .num_threads = 4,
    };

    // Allocate the largest vector buffer upfront
    const max_dataset_size = bench_config.vector_counts[bench_config.vector_counts.len - 1];
    try stdout.print("Allocating maximum dataset of {} vectors...\n", .{max_dataset_size});
    try stdout.flush();
    const vectors_buffer = try allocator.alignedAlloc(
        T,
        std.mem.Alignment.@"64",
        max_dataset_size * N,
    );
    defer allocator.free(vectors_buffer);

    // Fill with synthetic data
    var prng = std.Random.DefaultPrng.init(bench_config.seed);
    const random = prng.random();
    for (0..vectors_buffer.len) |i| {
        vectors_buffer[i] = random.float(T) * 100;
    }

    try stdout.print("Running benchmarks...\n\n", .{});
    try stdout.flush();

    var all_results = std.ArrayList(BenchmarkResult).empty;
    defer {
        for (all_results.items) |*result| {
            result.deinit(allocator);
        }
        all_results.deinit(allocator);
    }

    // Run benchmarks
    for (bench_config.vector_counts) |vector_count| {
        for (bench_config.graph_degrees) |graph_degree| {
            try stdout.print("Benchmarking: {} vectors, degree {}...\n", .{ vector_count, graph_degree });
            try stdout.flush();

            // Create a slice of the dataset
            const sliced_dataset = zagra.dataset.Dataset(T, N){
                .data_buffer = vectors_buffer[0 .. vector_count * N],
                .len = vector_count,
            };

            const result = try runBenchmark(
                T,
                N,
                sliced_dataset,
                graph_degree,
                bench_config,
                allocator,
            );
            try all_results.append(allocator, result);
        }
    }

    // Write results to CSV files
    const results_dir = csv.CSV_RESULTS_DIR;
    std.fs.cwd().access(results_dir, .{}) catch |e| switch (e) {
        error.FileNotFound => try std.fs.cwd().makeDir(results_dir),
        else => return e,
    };

    // Run summary
    const summary_name = "nn_descent_summary";
    const summary_file_name = if (result_prefix) |prefix|
        try std.fmt.allocPrint(allocator, "{s}/{s}_{s}.csv", .{ results_dir, prefix, summary_name })
    else
        try std.fmt.allocPrint(allocator, "{s}/{s}.csv", .{ results_dir, summary_name });
    const summary_csv_file = try std.fs.cwd().createFile(summary_file_name, .{});
    defer summary_csv_file.close();

    var summary_csv_buffer: [1024]u8 = undefined;
    var summary_csv_writer = summary_csv_file.writer(&summary_csv_buffer);
    const summary_csv = &summary_csv_writer.interface;

    const summary_headers = &[_][]const u8{
        "vector_count",
        "graph_degree",
        "total_training_s",
        "init_random_s",
        "num_iterations_completed",
        "converged",
    };
    try csv.writeHeaders(summary_csv, summary_headers);

    // Iteration details
    const iterations_name = "nn_descent_iterations";
    const iterations_file_name = if (result_prefix) |prefix|
        try std.fmt.allocPrint(allocator, "{s}/{s}_{s}.csv", .{ results_dir, prefix, iterations_name })
    else
        try std.fmt.allocPrint(allocator, "{s}/{s}.csv", .{ results_dir, iterations_name });
    const iterations_csv_file = try std.fs.cwd().createFile(iterations_file_name, .{});
    defer iterations_csv_file.close();

    var iterations_csv_buffer: [1024]u8 = undefined;
    var iterations_csv_writer = iterations_csv_file.writer(&iterations_csv_buffer);
    const iterations_csv = &iterations_csv_writer.interface;

    const iterations_headers = &[_][]const u8{
        "vector_count",
        "graph_degree",
        "iteration",
        "sample_s",
        "gen_s",
        "apply_s",
        "total_s",
        "updates",
    };
    try csv.writeHeaders(iterations_csv, iterations_headers);

    for (all_results.items) |result| {
        const train_timing = result.timing;
        try csv.writeRow(
            summary_csv,
            .{
                result.vector_count,
                result.graph_degree,
                @as(f64, @floatFromInt(train_timing.total_training_ns)) / std.time.ns_per_s,
                @as(f64, @floatFromInt(train_timing.init_random_ns)) / std.time.ns_per_s,
                train_timing.num_iterations_completed,
                if (train_timing.converged) "true" else "false",
            },
            summary_headers.len,
        );

        for (train_timing.iterations.items) |iter_timing| {
            try csv.writeRow(
                iterations_csv,
                .{
                    result.vector_count,
                    result.graph_degree,
                    iter_timing.iteration,
                    @as(f64, @floatFromInt(iter_timing.sample_candidates_ns)) / std.time.ns_per_s,
                    @as(f64, @floatFromInt(iter_timing.generate_proposals_ns)) / std.time.ns_per_s,
                    @as(f64, @floatFromInt(iter_timing.apply_updates_ns)) / std.time.ns_per_s,
                    @as(f64, @floatFromInt(iter_timing.total_iteration_ns)) / std.time.ns_per_s,
                    iter_timing.updates_count,
                },
                iterations_headers.len,
            );
        }
    }
    try summary_csv.flush();
    try iterations_csv.flush();

    try stdout.print(
        "Benchmarks completed.\nSummary written to {s}\nIteration details written to {s}\n",
        .{ summary_file_name, iterations_file_name },
    );
    try stdout.flush();
}
