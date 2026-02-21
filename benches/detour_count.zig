const std = @import("std");
const zagra = @import("zagra");
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

fn formatDuration(ns: u64, buf: []u8) []const u8 {
    if (ns >= 1_000_000_000) {
        const secs = @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
        return std.fmt.bufPrint(buf, "{d:.2}s", .{secs}) catch buf[0..0];
    } else if (ns >= 1_000_000) {
        const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
        return std.fmt.bufPrint(buf, "{d:.1}ms", .{ms}) catch buf[0..0];
    } else if (ns >= 1_000) {
        const us = @as(f64, @floatFromInt(ns)) / 1_000.0;
        return std.fmt.bufPrint(buf, "{d:.1}µs", .{us}) catch buf[0..0];
    } else {
        return std.fmt.bufPrint(buf, "{d}ns", .{ns}) catch buf[0..0];
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

    // Print results table
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    const col_width = 12;

    try stdout.print("\n╔══════════════", .{});
    for (ks_values) |_| try stdout.print("╦{s}", .{"═" ** col_width});
    try stdout.print("╗\n", .{});

    try stdout.print("║  N \\ K       ", .{});
    for (ks_values) |k| try stdout.print("║ K={d:<9}", .{k});
    try stdout.print("║\n", .{});

    try stdout.print("╠══════════════", .{});
    for (ks_values) |_| try stdout.print("╬{s}", .{"═" ** col_width});
    try stdout.print("╣\n", .{});

    for (ns_values, 0..) |num_nodes, n_idx| {
        try stdout.print("║ N={d:<10} ", .{num_nodes});
        for (ks_values, 0..) |_, k_idx| {
            var buf: [32]u8 = undefined;
            const dur_str = formatDuration(results[n_idx][k_idx], &buf);
            try stdout.print("║ {s:<10} ", .{dur_str});
        }
        try stdout.print("║\n", .{});

        if (n_idx < ns_values.len - 1) {
            try stdout.print("╠══════════════", .{});
            for (ks_values) |_| try stdout.print("╬{s}", .{"═" ** col_width});
            try stdout.print("╣\n", .{});
        }
    }

    try stdout.print("╚══════════════", .{});
    for (ks_values) |_| try stdout.print("╩{s}", .{"═" ** col_width});
    try stdout.print("╝\n", .{});
    try stdout.flush();
}
