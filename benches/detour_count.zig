const std = @import("std");
const zagra = @import("zagra");
const csv = @import("csv.zig");

const Optimizer = zagra.index.Optimizer;

fn fillRandomNeighbors(
    neighbors_list: *const Optimizer.NeighborsList,
    seed: u64,
    allocator: std.mem.Allocator,
) !void {
    const node_ids_random = try allocator.alloc(usize, neighbors_list.num_nodes);
    defer allocator.free(node_ids_random);
    for (node_ids_random, 0..) |*elem, node_id| {
        elem.* = node_id;
    }
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    random.shuffle(usize, node_ids_random);
    for (0..neighbors_list.num_nodes) |node_id| {
        const neighbors: []usize = neighbors_list.getEntryFieldSlice(node_id, .neighbor_id);
        var idx = node_id;
        for (0..neighbors_list.num_neighbors_per_node) |neighbor_idx| {
            var neighbor_id = node_ids_random[idx % node_ids_random.len];
            if (neighbor_id == node_id) {
                idx += 1;
                neighbor_id = node_ids_random[idx % node_ids_random.len];
            }
            idx += 1;
            neighbors[neighbor_idx] = neighbor_id;
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const ns_values = [_]usize{ 1_000_000, 5_000_000, 10_000_000 };
    const ks_values = [_]usize{ 8, 16, 32, 64, 128 };
    const seed: u64 = 42;

    // results[n_idx][k_idx]
    var results: [ns_values.len][ks_values.len]u64 = undefined;

    var thread_pool: std.Thread.Pool = undefined;
    try thread_pool.init(.{ .allocator = allocator });
    defer thread_pool.deinit();

    for (ns_values, 0..) |num_nodes, n_idx| {
        for (ks_values, 0..) |num_neighbors_per_node, k_idx| {
            std.debug.print("Benchmarking N={d}, K={d}...\n", .{ num_nodes, num_neighbors_per_node });

            var neighbors_list = try Optimizer.NeighborsList.init(
                num_nodes,
                num_neighbors_per_node,
                allocator,
            );
            defer neighbors_list.deinit(allocator);

            try fillRandomNeighbors(&neighbors_list, seed, allocator);

            var optimizer = Optimizer.init(
                neighbors_list,
                &thread_pool,
                neighbors_list.num_nodes,
            );

            var timer = try std.time.Timer.start();
            optimizer.countDetours();
            results[n_idx][k_idx] = timer.read();
        }
    }

    const results_dir = "benches/results";
    std.fs.cwd().access(results_dir, .{}) catch |e| switch (e) {
        error.FileNotFound => try std.fs.cwd().makeDir(results_dir),
        else => return e,
    };

    const file_name = results_dir ++ "/detour_count.csv";
    const csv_file = try std.fs.cwd().createFile(file_name, .{});
    defer csv_file.close();

    std.debug.print("Writing results to {s}...\n", .{file_name});

    var csv_buffer: [1024]u8 = undefined;
    var csv_writer = csv_file.writer(&csv_buffer);
    const writer = &csv_writer.interface;

    const headers = &[_][]const u8{ "n", "k", "time_ns" };
    try csv.writeHeaders(writer, headers);
    for (ns_values, 0..) |num_nodes, n_idx| {
        for (ks_values, 0..) |k_val, k_idx| {
            const row = .{ num_nodes, k_val, results[n_idx][k_idx] };
            try csv.writeRow(writer, row);
        }
    }
    try writer.flush();
    std.debug.print("Results written to {s}\n", .{file_name});
}
