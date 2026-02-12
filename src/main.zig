const std = @import("std");

const znpy = @import("znpy");
const zagra = @import("zagra");

pub const std_options: std.Options = .{
    .log_level = .info,
};

var stdout_buffer: [1024]u8 = undefined;
var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
const stdout = &stdout_writer.interface;

const HELP =
    \\zagra <vector_count> <graph_degree> [block_processing]
    \\- vector_count (required): Number of vectors in the dataset
    \\- graph_degree (required): Graph degree of vectors in the KNN graph
    \\- block_processing (optional - default to true): Whether to use block processing mode during training
;

pub fn main() !void {
    std.debug.print("This is Zagra!\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.skip();

    const vector_count_str = args.next() orelse {
        std.debug.print("vector count needed.\n{s}", .{HELP});
        return;
    };
    const vector_count = std.fmt.parseInt(usize, vector_count_str, 10) catch |e| {
        switch (e) {
            error.Overflow => std.debug.print("Entered vector count is too large for usize\n", .{}),
            error.InvalidCharacter => std.debug.print("Entered vector count is not a valid number\n", .{}),
        }
        return;
    };

    const graph_degree_str = args.next() orelse {
        std.debug.print("graph degree needed.\n{s}", .{HELP});
        return;
    };
    const graph_degree = std.fmt.parseInt(usize, graph_degree_str, 10) catch |e| {
        switch (e) {
            error.Overflow => std.debug.print("Entered graph degree is too large for usize\n", .{}),
            error.InvalidCharacter => std.debug.print("Entered graph degree is not a valid number\n", .{}),
        }
        return;
    };
    try stdout.print("Number of neighbors per vector: {any}\n", .{graph_degree});

    const block_processing = blk: {
        if (args.next()) |str| {
            if (std.mem.eql(u8, str, "true"))
                break :blk true
            else if (std.mem.eql(u8, str, "false"))
                break :blk false
            else {
                std.debug.print("Expected 'true' or 'false' for block_processing", .{});
                return;
            }
        } else break :blk true;
    };
    try stdout.print("Using block processing for training? {any}\n", .{block_processing});

    // Dataset configuration constants
    const T = i32; // Element type
    const N: usize = 128; // Vector length

    try stdout.print("Creating dataset with {} {}-D vectors\n", .{ vector_count, N });
    try stdout.flush();

    // Allocate aligned buffer for vectors
    const vectors_buffer = allocator.alignedAlloc(
        T,
        std.mem.Alignment.@"64",
        vector_count * N,
    ) catch {
        std.debug.print("Dataset size too large. Please try smaller number of vectors.\n", .{});
        return;
    };
    defer allocator.free(vectors_buffer);

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    for (0..vectors_buffer.len) |i| {
        vectors_buffer[i] = @rem(rng.int(T), 2);
    }

    // Create dataset from buffer
    const Dataset = zagra.dataset.Dataset(T, N);
    const dataset = Dataset{
        .data_buffer = vectors_buffer,
        .len = vector_count,
    };

    std.debug.assert(dataset.len == vector_count);

    // Do NN-Descent
    const NNDescent = zagra.index.nn_descent.NNDescent(T, N);
    var training_config = zagra.index.nn_descent.TrainingConfig.init(
        4,
        vector_count,
        null,
        42,
    );
    training_config.block_processing = block_processing;
    training_config.num_neighbors_per_node = graph_degree;

    var nn_descent = NNDescent.init(
        dataset,
        training_config,
        allocator,
    ) catch |e| {
        std.debug.print("Error initializing NNDescent: {}", .{e});
        return;
    };
    defer nn_descent.deinit(allocator);

    try stdout.print("Start timing NN-Descent training...\n", .{});
    try stdout.flush();

    var timer = try std.time.Timer.start();
    nn_descent.train();
    const elapsed_time_ns = timer.read();
    const elapsed_time_s: f64 = @as(f64, @floatFromInt(elapsed_time_ns)) / 1_000_000_000.0;

    try stdout.print("Training for {} vectors with graph degree of {} took: {}s\n", .{ dataset.len, graph_degree, elapsed_time_s });
    try stdout.flush();

    // Print neighbors of the first few vector
    for (0..5) |node_id| {
        if (node_id >= dataset.len) break;
        const neighbor_ids: []const isize = nn_descent.neighbors_list.getEntryFieldSlice(node_id, .neighbor_id);
        const neighbor_distances: []const T = nn_descent.neighbors_list.getEntryFieldSlice(node_id, .distance);
        try stdout.print("Neighbors of node {}:\n", .{node_id});
        for (neighbor_ids, 0..) |neighbor_id, i| {
            const distance = neighbor_distances[i];
            try stdout.print("Neighbor {}: ID={}, Distance={}\n", .{ i, neighbor_id, distance });
        }
    }
    try stdout.flush();
}
