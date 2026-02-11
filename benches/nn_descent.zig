const std = @import("std");
const znpy = @import("znpy");
const zagra = @import("zagra");

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
    block_processing: bool = true,
};

const BenchmarkResult = struct {
    vector_count: usize,
    graph_degree: usize,
    timing: zagra.graphs.nn_descent.TrainingTiming,

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
    const NNDescent = zagra.graphs.nn_descent.NNDescent(T, N);
    var training_config = zagra.graphs.nn_descent.TrainingConfig.init(
        config.num_threads,
        dataset.len,
        null,
        config.seed,
    );
    training_config.block_processing = config.block_processing;
    training_config.num_neighbors_per_node = graph_degree;
    training_config.max_iterations = config.max_iterations;
    training_config.delta = config.delta;

    var nn_descent = try NNDescent.init(dataset, training_config, allocator);
    defer nn_descent.deinit(allocator);

    const timing = try nn_descent.trainWithTiming(allocator);

    return BenchmarkResult{
        .vector_count = dataset.len,
        .graph_degree = graph_degree,
        .timing = timing,
    };
}

fn printResults(results: []BenchmarkResult, writer: *std.io.Writer) !void {
    try writer.print("\n", .{});
    try writer.print("=" ** 80 ++ "\n", .{});
    try writer.print("NN-DESCENT BENCHMARK RESULTS\n", .{});
    try writer.print("=" ** 80 ++ "\n\n", .{});

    for (results) |result| {
        const train_timing = result.timing;
        try writer.print("Vectors: {d:>8} | Degree: {d:>3} | Total: {d:>8.3}s | Iters: {d:>2} | Converged: {}\n", .{
            result.vector_count,
            result.graph_degree,
            @as(f64, @floatFromInt(train_timing.total_training_ns)) / 1_000_000_000.0,
            train_timing.num_iterations_completed,
            train_timing.converged,
        });

        try writer.print("  Init random neighbors: {d:>8.3}s\n", .{
            @as(f64, @floatFromInt(train_timing.init_random_ns)) / 1_000_000_000.0,
        });

        try writer.print("  Iteration breakdown:\n", .{});
        try writer.print("    {s:>4} | {s:>10} | {s:>10} | {s:>10} | {s:>10} | {s:>8}\n", .{ "Iter", "Sample (s)", "Gen (s)", "Apply (s)", "Total (s)", "Updates" });
        try writer.print("    " ++ "-" ** 70 ++ "\n", .{});

        for (train_timing.iterations.items) |iter_timing| {
            try writer.print("    {d:>4} | {d:>10.3} | {d:>10.3} | {d:>10.3} | {d:>10.3} | {d:>8}\n", .{
                iter_timing.iteration,
                @as(f64, @floatFromInt(iter_timing.sample_candidates_ns)) / 1_000_000_000.0,
                @as(f64, @floatFromInt(iter_timing.generate_proposals_ns)) / 1_000_000_000.0,
                @as(f64, @floatFromInt(iter_timing.apply_updates_ns)) / 1_000_000_000.0,
                @as(f64, @floatFromInt(iter_timing.total_iteration_ns)) / 1_000_000_000.0,
                iter_timing.updates_count,
            });
        }
        try writer.print("\n", .{});
    }

    try writer.print("=" ** 80 ++ "\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const T: type = f32; // Element type
    const N: usize = 128; // Vector length

    // Benchmark configuration
    const bench_config = BenchmarkConfig{
        .vector_counts = &[_]usize{ 1_000_000, 5_000_000, 10_000_000 },
        .graph_degrees = &[_]usize{ 8, 16, 32, 64, 128 },
        .num_threads = 4,
        .block_processing = true,
    };

    // Allocate the largest vector buffer upfront
    const max_dataset_size = bench_config.vector_counts[bench_config.vector_counts.len - 1];
    std.debug.print("Allocating maximum dataset of {} vectors...\n", .{max_dataset_size});
    const vectors_buffer = try allocator.alignedAlloc(
        T,
        std.mem.Alignment.@"64",
        max_dataset_size * N,
    );
    defer allocator.free(vectors_buffer);
    // Fill with synthetic data
    var prng = std.Random.DefaultPrng.init(bench_config.seed);
    const rng = prng.random();
    for (0..vectors_buffer.len) |i| {
        vectors_buffer[i] = rng.float(T) * 100;
    }
    std.debug.print("Running benchmarks...\n\n", .{});

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
            std.debug.print("Benchmarking: {} vectors, degree {}...\n", .{ vector_count, graph_degree });

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

    // Print results
    var write_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&write_buffer);
    const stdout = &stdout_writer.interface;
    try printResults(all_results.items, stdout);
    try stdout.flush();
}
