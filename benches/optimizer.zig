const std = @import("std");
const znpy = @import("znpy");
const zagra = @import("zagra");
const csv = @import("csv.zig");

pub const std_options: std.Options = .{
    .log_level = .err,
};

const BenchmarkConfig = struct {
    vector_counts: []const usize,
    output_graph_degrees: []const usize,
    intermediate_degree_multiplier: usize = 2,
    num_threads: usize = 4,
    nn_descent_max_iterations: usize = 10,
    nn_descent_delta: f32 = 0.001,
    seed: u64 = 42,
};

const BenchmarkResult = struct {
    vector_count: usize,
    output_degree: usize,
    intermediate_degree: usize,
    timing: zagra.index.Optimizer.OptimizationTiming,
    graph: zagra.index.Optimizer.NeighborsList(false),

    fn deinit(self: *BenchmarkResult, allocator: std.mem.Allocator) void {
        self.graph.deinit(allocator);
    }
};

fn runBenchmark(
    comptime T: type,
    comptime N: usize,
    dataset: zagra.dataset.Dataset(T, N),
    output_degree: usize,
    config: BenchmarkConfig,
    allocator: std.mem.Allocator,
) !BenchmarkResult {
    const intermediate_degree = output_degree * config.intermediate_degree_multiplier;

    const NNDescent = zagra.index.NNDescent(T, N);
    var training_config = zagra.index.TrainingConfig.init(
        intermediate_degree,
        dataset.len,
        null,
        config.seed,
    );
    training_config.max_iterations = config.nn_descent_max_iterations;
    training_config.delta = config.nn_descent_delta;

    var nn_descent = try NNDescent.init(dataset, training_config, allocator);
    defer nn_descent.deinit(allocator);

    nn_descent.train();
    nn_descent.sortNeighbors();

    const neighbor_entries = nn_descent.neighbors_list.entries;
    const num_nodes = nn_descent.neighbors_list.num_nodes;
    const num_neighbors_per_node = nn_descent.neighbors_list.num_neighbors_per_node;

    const neighbor_ids: []usize = try allocator.dupe(usize, neighbor_entries.items(.neighbor_id));
    defer allocator.free(neighbor_ids);

    const detour_counts: []usize = try allocator.alloc(usize, neighbor_entries.len);
    defer allocator.free(detour_counts);

    const optimizer_entries = zagra.index.SoaSlice(zagra.index.Optimizer.NeighborsList(true).Entry){
        .ptrs = [_][*]u8{
            @ptrCast(neighbor_ids.ptr),
            @ptrCast(detour_counts.ptr),
        },
        .len = neighbor_entries.len,
    };

    var optimizer = zagra.index.Optimizer.init(
        zagra.index.Optimizer.NeighborsList(true){
            .entries = optimizer_entries,
            .num_neighbors_per_node = num_neighbors_per_node,
            .num_nodes = num_nodes,
        },
        null,
        nn_descent.num_nodes_per_block,
    );

    const result = try optimizer.optimizeWithTiming(output_degree, allocator);

    return BenchmarkResult{
        .vector_count = dataset.len,
        .output_degree = output_degree,
        .intermediate_degree = intermediate_degree,
        .timing = result.timing,
        .graph = result.graph,
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const result_name = "optimizer_summary";

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.skip();

    const result_prefix = args.next();

    const T: type = f32;
    const N: usize = 128;

    const bench_config = BenchmarkConfig{
        .vector_counts = &[_]usize{ 100_000, 500_000 },
        .output_graph_degrees = &[_]usize{ 8, 16, 32, 64 },
        .num_threads = 4,
    };

    const max_dataset_size = bench_config.vector_counts[bench_config.vector_counts.len - 1];
    std.debug.print("Allocating maximum dataset of {} vectors...\n", .{max_dataset_size});
    const vectors_buffer = try allocator.alignedAlloc(
        T,
        std.mem.Alignment.@"64",
        max_dataset_size * N,
    );
    defer allocator.free(vectors_buffer);

    var prng = std.Random.DefaultPrng.init(bench_config.seed);
    const random = prng.random();
    for (0..vectors_buffer.len) |i| {
        vectors_buffer[i] = random.float(T) * 100;
    }
    std.debug.print("Running benchmarks...\n\n", .{});

    var all_results = std.ArrayList(BenchmarkResult).empty;
    defer {
        for (all_results.items) |*result| {
            result.deinit(allocator);
        }
        all_results.deinit(allocator);
    }

    for (bench_config.vector_counts) |vector_count| {
        for (bench_config.output_graph_degrees) |output_degree| {
            std.debug.print("Benchmarking: {} vectors, output degree {}...\n", .{ vector_count, output_degree });

            const sliced_dataset = zagra.dataset.Dataset(T, N){
                .data_buffer = vectors_buffer[0 .. vector_count * N],
                .len = vector_count,
            };

            const result = try runBenchmark(
                T,
                N,
                sliced_dataset,
                output_degree,
                bench_config,
                allocator,
            );
            try all_results.append(allocator, result);
        }
    }

    const results_dir = "benches/results";
    std.fs.cwd().access(results_dir, .{}) catch |e| switch (e) {
        error.FileNotFound => try std.fs.cwd().makeDir(results_dir),
        else => return e,
    };

    const summary_file_name = if (result_prefix) |prefix|
        try std.fmt.allocPrint(allocator, "{s}/{s}_{s}.csv", .{ results_dir, prefix, result_name })
    else
        try std.fmt.allocPrint(allocator, "{s}/{s}.csv", .{ results_dir, result_name });
    const summary_csv_file = try std.fs.cwd().createFile(summary_file_name, .{});
    defer summary_csv_file.close();
    var summary_csv_buffer: [1024]u8 = undefined;
    var summary_csv_writer = summary_csv_file.writer(&summary_csv_buffer);
    const summary_csv = &summary_csv_writer.interface;

    try csv.writeHeaders(summary_csv, &[_][]const u8{
        "vector_count", "output_degree", "intermediate_degree", "total_s", "detour_count_s", "prune_s", "reverse_graph_s", "combine_s",
    });

    for (all_results.items) |result| {
        try csv.writeRow(summary_csv, .{
            result.vector_count,
            result.output_degree,
            result.intermediate_degree,
            @as(f64, @floatFromInt(result.timing.total_optimization_ns)) / 1_000_000_000.0,
            @as(f64, @floatFromInt(result.timing.count_detours_ns)) / 1_000_000_000.0,
            @as(f64, @floatFromInt(result.timing.prune_ns)) / 1_000_000_000.0,
            @as(f64, @floatFromInt(result.timing.build_reverse_graph_ns)) / 1_000_000_000.0,
            @as(f64, @floatFromInt(result.timing.combine_ns)) / 1_000_000_000.0,
        });
    }
    try summary_csv.flush();

    std.debug.print("Benchmarks completed.\nSummary written to {s}\n", .{summary_file_name});
}
