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
    \\zagra <vector_count> <graph_degree> [intermediate_graph_degree] [options]
    \\- vector_count (required): Number of vectors in the dataset
    \\- graph_degree (required): Graph degree of the final CAGRA graph
    \\Options:
    \\- --threads <n>: Number of threads for NN-Descent and Optimizer (default: CPU core count)
    \\- --block-size <n>: Block size for processing (default: 16384)
    \\- --save <path> or -o <path>: Save the built index to the specified directory (optional)
;

pub fn main() !void {
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

    // Default values
    const intermediate_graph_degree: usize = graph_degree * 2;
    var num_threads: ?usize = null;
    var block_size: usize = 16384;
    var save_path: ?[]const u8 = null;

    // Parse optional flags
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--threads")) {
            const threads_str = args.next() orelse {
                std.debug.print("--threads requires a value\n", .{});
                return;
            };
            num_threads = std.fmt.parseInt(usize, threads_str, 10) catch |e| {
                switch (e) {
                    error.Overflow => std.debug.print("Entered thread count is too large for usize\n", .{}),
                    error.InvalidCharacter => std.debug.print("Entered thread count is not a valid number\n", .{}),
                }
                return;
            };
        } else if (std.mem.eql(u8, arg, "--block-size")) {
            const bs_str = args.next() orelse {
                std.debug.print("--block-size requires a value\n", .{});
                return;
            };
            block_size = std.fmt.parseInt(usize, bs_str, 10) catch |e| {
                switch (e) {
                    error.Overflow => std.debug.print("Entered block size is too large for usize\n", .{}),
                    error.InvalidCharacter => std.debug.print("Entered block size is not a valid number\n", .{}),
                }
                return;
            };
        } else if (std.mem.eql(u8, arg, "--save") or std.mem.eql(u8, arg, "-o")) {
            save_path = args.next() orelse {
                std.debug.print("--save requires a path\n", .{});
                return;
            };
        } else {
            std.debug.print("Unknown argument: {s}\n{s}", .{ arg, HELP });
            return;
        }
    }

    try stdout.print("=== Zagra Index Build ===\n", .{});
    try stdout.print("Number of vectors: {}\n", .{vector_count});
    try stdout.print("Vector dimensions: 128\n", .{});
    try stdout.print("Graph degree: {}\n", .{graph_degree});
    try stdout.print("Intermediate graph degree: {}\n", .{intermediate_graph_degree});
    if (num_threads) |t| {
        try stdout.print("Number of threads: {}\n", .{t});
    } else {
        const cpu_count = std.Thread.getCpuCount() catch 1;
        try stdout.print("Number of threads: {} (default)\n", .{cpu_count});
    }
    try stdout.print("Block size: {}\n", .{block_size});
    try stdout.flush();

    // Dataset configuration constants
    const T = i32; // Element type
    const N: usize = 128; // Vector length

    // Allocate aligned buffer for vectors
    const vectors_buffer = allocator.alignedAlloc(
        T,
        std.mem.Alignment.@"64",
        vector_count * N,
    ) catch {
        std.debug.print("Dataset size too large. Please try smaller number of vectors.\n", .{});
        return;
    };
    errdefer allocator.free(vectors_buffer);

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (0..vectors_buffer.len) |i| {
        vectors_buffer[i] = @rem(random.int(T), 2);
    }

    // Create dataset from buffer
    const Dataset = zagra.dataset.Dataset(T, N);
    const dataset = Dataset{
        .data_buffer = vectors_buffer,
        .len = vector_count,
    };

    std.debug.assert(dataset.len == vector_count);

    // Build config
    const build_config = zagra.index.BuildConfig.init(
        graph_degree,
        intermediate_graph_degree,
        vector_count,
        num_threads,
        42,
        block_size,
    );

    // Build index
    const Index = zagra.index.Index(T, N);

    try stdout.print("\nBuilding index...\n", .{});
    try stdout.flush();

    var timer = try std.time.Timer.start();
    var index = try Index.build(dataset, build_config, allocator);
    defer index.deinit(allocator);
    const build_time_ns = timer.read();
    const build_time_s: f64 = @as(f64, @floatFromInt(build_time_ns)) / 1_000_000_000.0;

    try stdout.print("\n=== Build Complete ===\n", .{});
    try stdout.print("Total build time: {:.3}s\n", .{build_time_s});
    try stdout.print("\n=== Index Info ===\n", .{});
    try stdout.print("Number of nodes: {}\n", .{index.num_nodes});
    try stdout.print("Graph degree: {}\n", .{index.num_neighbors_per_node});
    try stdout.print("Total edges: {}\n", .{index.num_nodes * index.num_neighbors_per_node});
    try stdout.flush();

    // Print neighbors of the first few vectors
    try stdout.print("\n=== Sample Neighbors ===\n", .{});
    const sample_count: usize = 3;
    for (0..sample_count) |node_id| {
        if (node_id >= index.num_nodes) break;
        const neighbor_slice = index.graph[node_id * index.num_neighbors_per_node ..][0..index.num_neighbors_per_node];
        try stdout.print("Node {}: ", .{node_id});
        for (neighbor_slice) |neighbor_id| {
            try stdout.print("{} ", .{neighbor_id});
        }
        try stdout.print("\n", .{});
        try stdout.flush();
    }

    // Save the index to disk if path is provided
    if (save_path) |path| {
        try stdout.print("Saving index to {s}\n", .{path});
        try stdout.flush();
        try index.save(path, allocator);
        try stdout.print("Index saved.\n", .{});
        try stdout.flush();

        // Load the index back from disk to verify
        try stdout.print("Loading index from {s}\n", .{path});
        try stdout.flush();
        var loaded_index = try Index.load(path, allocator);
        defer loaded_index.deinit(allocator);
        try stdout.print("Index loaded. Verifying...\n", .{});
        try stdout.flush();

        const graph_match = loaded_index.num_nodes == index.num_nodes and loaded_index.num_neighbors_per_node == index.num_neighbors_per_node and std.mem.eql(usize, loaded_index.graph, index.graph);
        const dataset_match = loaded_index.dataset.len == index.dataset.len and std.mem.eql(T, loaded_index.dataset.data_buffer, index.dataset.data_buffer);
        if (graph_match and dataset_match) {
            try stdout.print("Loaded index matches original index.\n", .{});
        } else {
            try stdout.print("Loaded index does not match original index!\n", .{});
        }
        try stdout.flush();
    }
}
