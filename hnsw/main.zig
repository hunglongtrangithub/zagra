const std = @import("std");
const root = @import("root.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const dim: usize = 8;
    const maxElements: usize = 1024;

    std.debug.print("=== Creating indexes ===\n", .{});

    var bf_index = try root.Bruteforce.create(dim, maxElements);
    defer bf_index.deinit();

    var hnsw_index = try root.HierarchicalIndex.create(
        dim,
        maxElements,
        16,
        200,
        42,
        false,
    );
    defer hnsw_index.deinit();

    std.debug.print("Created Bruteforce and HNSW indexes (dim={}, max_elements={})\n", .{ dim, maxElements });

    var points: [3][dim]f32 = .{
        .{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        .{ 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        .{ 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 },
    };

    try bf_index.addPoint(points[0][0..], 1);
    try bf_index.addPoint(points[1][0..], 2);
    try bf_index.addPoint(points[2][0..], 3);

    try hnsw_index.addPoint(points[0][0..], 1, false);
    try hnsw_index.addPoint(points[1][0..], 2, false);
    try hnsw_index.addPoint(points[2][0..], 3, false);

    std.debug.print("Added 3 points (labels 1,2,3) to both indexes\n", .{});

    var query: [dim]f32 = .{ 1.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const k: usize = 2;

    std.debug.print("\n=== Searching with query (1.05, 0, 0, ...) k={} ===\n", .{k});

    var bf_result = try bf_index.searchKnnAlloc(allocator, query[0..], k);
    defer bf_result.deinit(allocator);

    var hnsw_result = try hnsw_index.searchKnnAlloc(allocator, query[0..], k);
    defer hnsw_result.deinit(allocator);

    std.debug.print("Bruteforce results ({} found):\n", .{bf_result.len});
    for (bf_result.items(.label), bf_result.items(.distance)) |label, dist| {
        std.debug.print("  label={}, dist={}\n", .{ label, dist });
    }

    std.debug.print("HNSW results ({} found):\n", .{hnsw_result.len});
    for (hnsw_result.items(.label), hnsw_result.items(.distance)) |label, dist| {
        std.debug.print("  label={}, dist={}\n", .{ label, dist });
    }

    try bf_index.removePoint(2);
    try hnsw_index.markDelete(2);
    std.debug.print("\n=== Removed label 2 from both indexes ===\n", .{});

    var bf_result2 = try bf_index.searchKnnAlloc(allocator, query[0..], k);
    defer bf_result2.deinit(allocator);

    var hnsw_result2 = try hnsw_index.searchKnnAlloc(allocator, query[0..], k);
    defer hnsw_result2.deinit(allocator);

    std.debug.print("Bruteforce after removal ({} found):\n", .{bf_result2.len});
    for (bf_result2.items(.label), bf_result2.items(.distance)) |label, dist| {
        std.debug.print("  label={}, dist={}\n", .{ label, dist });
    }

    std.debug.print("HNSW after removal ({} found):\n", .{hnsw_result2.len});
    for (hnsw_result2.items(.label), hnsw_result2.items(.distance)) |label, dist| {
        std.debug.print("  label={}, dist={}\n", .{ label, dist });
    }

    std.debug.print("\nExample complete.\n", .{});
}
