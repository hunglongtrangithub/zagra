const std = @import("std");
const builtin = @import("builtin");
const log = std.log.scoped(.index);

const znpy = @import("znpy");

const mod_types = @import("types.zig");
const mod_dataset = @import("dataset.zig");
const mod_soa_slice = @import("index/soa_slice.zig");
const mod_optimizer = @import("index/optimizer.zig");
const mod_nn_descent = @import("index/nn_descent.zig");
const mod_searcher = @import("index/searcher.zig");

pub const NNDescent = mod_nn_descent.NNDescent;
pub const NNDTrainingConfig = mod_nn_descent.TrainingConfig;
pub const NNDTrainingTiming = mod_nn_descent.TrainingTiming;
pub const SoaSlice = mod_soa_slice.SoaSlice;
pub const Optimizer = mod_optimizer.Optimizer;
pub const SearchConfig = mod_searcher.SearchConfig;

/// Check that the graph is valid:
/// - The length of neighbor_ids must equal number of nodes * number of neighbors per node
/// - All neighbor IDs must be in the range [0, num_nodes)
/// - No node can have itself as a neighbor
/// - No duplicate neighbor IDs for the same node
pub fn isValidGraph(
    neighbor_ids: []const usize,
    num_nodes: usize,
    num_neighbors_per_node: usize,
) bool {
    if (neighbor_ids.len != num_nodes * num_neighbors_per_node) {
        log.err(
            "Graph length {} does not match expected size {}",
            .{ neighbor_ids.len, num_nodes * num_neighbors_per_node },
        );
        return false;
    }

    for (0..num_nodes) |node_id| {
        const start = node_id * num_neighbors_per_node;
        const end = start + num_neighbors_per_node;
        const slice = neighbor_ids[start..end];

        for (slice, 0..) |neighbor_id, neighbor_idx| {
            if (neighbor_id >= num_nodes) {
                log.err(
                    "Invalid neighbor ID {} found for node {} in neighbor IDs {any}",
                    .{ neighbor_id, node_id, slice },
                );
                return false;
            }

            if (neighbor_id == node_id) {
                log.err(
                    "Node {} has itself as a neighbor in neighbor IDs {any}",
                    .{ node_id, slice },
                );
                return false;
            }

            for (slice[0..neighbor_idx]) |prev_neighbor_id| {
                if (neighbor_id == prev_neighbor_id) {
                    log.err(
                        "Duplicate neighbor ID {} found for node {} in neighbor IDs {any}",
                        .{ neighbor_id, node_id, slice },
                    );
                    return false;
                }
            }
        }
    }

    return true;
}

/// Configuration for building the index.
/// This includes parameters for both the initial graph construction and the subsequent optimization step.
pub const BuildConfig = struct {
    graph_degree: usize,
    nn_descent_config: mod_nn_descent.TrainingConfig,

    const Self = @This();

    pub fn init(
        graph_degree: usize,
        intermediate_graph_degree: usize,
        num_vectors: usize,
        num_threads: ?usize,
        seed: ?u64,
        block_size: usize,
    ) Self {
        const nn_descent_config = NNDTrainingConfig.init(
            intermediate_graph_degree,
            num_vectors,
            num_threads,
            seed,
        );
        var config = Self{
            .graph_degree = graph_degree,
            .nn_descent_config = nn_descent_config,
        };
        config.nn_descent_config.block_size = block_size;
        return config;
    }
};

pub fn Index(comptime T: type, comptime N: usize) type {
    return struct {
        /// The dataset of vectors. Owned by this struct.
        dataset: Dataset,
        /// The k-NN graph. Owned by this struct.
        graph: []const usize,
        /// Number of nodes (same as `dataset.len`).
        num_nodes: usize,
        /// Number of neighbors per node (graph degree).
        num_neighbors_per_node: usize,

        const NeighborsList = mod_optimizer.Optimizer.NeighborsList;

        pub const Dataset = mod_dataset.Dataset(T, N);
        pub const Searcher = mod_searcher.Searcher(T, N);
        pub const DATASET_FILENAME = "dataset.npy";
        pub const GRAPH_FILENAME = "graph.npy";

        const Self = @This();

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.graph);
            self.dataset.deinit(allocator);
        }

        /// Loads the dataset from a .npy file reader.
        /// Validates that the shape has 2 dimensions and the second dimension equals N.
        fn loadDataset(
            reader: *std.io.Reader,
            allocator: std.mem.Allocator,
        ) !Dataset {
            const array = try znpy.array.static.StaticArray(T, 2).fromFileAllocAligned(
                reader,
                std.mem.Alignment.@"64",
                allocator,
            );

            const shape = array.shape.dims;
            // Second dimension must equal N (vector length)
            if (shape[1] != N) {
                return error.InvalidDatasetDimension;
            }

            return Dataset{
                .data_buffer = @as([]align(64) const T, @alignCast(array.data_buffer)),
                .len = shape[0],
            };
        }

        /// Loads the graph from a .npy file reader.
        /// Validates that:
        ///   - The shape has 2 dimensions
        ///   - The first dimension equals `expected_num_nodes`
        ///   - The dtype is `u32` or `u64` and fits in `usize`
        ///   - All neighbor IDs are in valid range
        /// Return the graph as a slice of `usize` neighbor IDs in row-major order (length = num_nodes * degree),
        /// and the graph degree (number of neighbors per node).
        fn loadGraph(
            reader: *std.io.Reader,
            allocator: std.mem.Allocator,
            expected_num_nodes: usize,
        ) !struct { []usize, usize } {
            // Read header first to get dtype
            const header = try znpy.header.Header.fromReader(reader, allocator);
            defer header.deinit(allocator);
            const shape = header.shape;

            // Validate shape
            if (shape.len != 2) {
                return error.InvalidGraphShape;
            }
            const num_nodes = shape[0];
            const degree = shape[1];
            const num_elements = num_nodes * degree;

            if (num_nodes != expected_num_nodes) {
                return error.GraphDatasetMismatch;
            }

            // Check dtype
            const dtype_size: usize = switch (header.descr) {
                .UInt8 => @sizeOf(u8),
                .UInt16 => @sizeOf(u16),
                .UInt32 => @sizeOf(u32),
                .UInt64 => @sizeOf(u64),
                else => return error.InvalidGraphDtype,
            };
            // usize must be large enough to hold the neighbor ID
            if (dtype_size > @sizeOf(usize)) {
                return error.GraphDtypeTooLarge;
            }

            log.debug("loadGraph: num_nodes={}, degree={}, num_elements={}, dtype_size={}", .{
                num_nodes, degree, num_elements, dtype_size,
            });

            // Read and parse based on dtype
            const graph_data = try allocator.alloc(usize, num_elements);
            switch (header.descr) {
                .UInt8 => {
                    const typed_data = try allocator.alloc(u8, num_elements);
                    defer allocator.free(typed_data);
                    try reader.readSliceAll(std.mem.sliceAsBytes(typed_data));
                    for (typed_data, 0..) |v, i| {
                        graph_data[i] = @intCast(v);
                    }
                },
                .UInt16 => {
                    const typed_data = try allocator.alloc(u16, num_elements);
                    defer allocator.free(typed_data);
                    try reader.readSliceAll(std.mem.sliceAsBytes(typed_data));
                    for (typed_data, 0..) |v, i| {
                        graph_data[i] = @intCast(v);
                    }
                },
                .UInt32 => {
                    const typed_data = try allocator.alloc(u32, num_elements);
                    defer allocator.free(typed_data);
                    try reader.readSliceAll(std.mem.sliceAsBytes(typed_data));
                    for (typed_data, 0..) |v, i| {
                        graph_data[i] = @intCast(v);
                    }
                },
                .UInt64 => {
                    const typed_data = try allocator.alloc(u64, num_elements);
                    defer allocator.free(typed_data);
                    try reader.readSliceAll(std.mem.sliceAsBytes(typed_data));
                    for (typed_data, 0..) |v, i| {
                        graph_data[i] = @intCast(v);
                    }
                },
                else => unreachable,
            }

            // Validate all neighbor IDs are in range
            for (graph_data) |neighbor_id| {
                if (neighbor_id >= expected_num_nodes) {
                    return error.InvalidNeighborId;
                }
            }

            return .{ graph_data, degree };
        }

        /// Loads an index from a directory.
        ///
        /// Expected files in the directory:
        ///   - `dataset.npy`: The vector data (shape: (num_vectors, N), dtype: T)
        ///   - `graph.npy`: The k-NN graph (shape: (num_nodes, degree), dtype: u32 or u64)
        ///
        /// Validates that:
        ///   - The graph has 2 dimensions
        ///   - The first dimension of graph equals the number of vectors in dataset
        ///   - All neighbor IDs in the graph are valid (within [0, num_nodes))
        ///
        /// Returns an error if validation fails.
        pub fn load(
            dir_path: []const u8,
            allocator: std.mem.Allocator,
        ) !Self {
            var read_buffer: [4096]u8 = undefined;

            // Validate directory exists
            var dir = try std.fs.cwd().openDir(dir_path, .{});
            defer dir.close();

            // Load dataset.npy
            const dataset_file = try dir.openFile(DATASET_FILENAME, .{});
            defer dataset_file.close();

            var dataset_reader = dataset_file.reader(&read_buffer);
            const dataset = try loadDataset(&dataset_reader.interface, allocator);
            errdefer dataset.deinit(allocator);

            // Load graph.npy
            const graph_file = try dir.openFile(GRAPH_FILENAME, .{});
            defer graph_file.close();

            var graph_reader = graph_file.reader(&read_buffer);
            const graph_data, const graph_degree = try loadGraph(
                &graph_reader.interface,
                allocator,
                dataset.len,
            );

            return Self{
                .dataset = dataset,
                .graph = graph_data,
                .num_nodes = dataset.len,
                .num_neighbors_per_node = graph_degree,
            };
        }

        /// Saves the index to a directory.
        ///
        /// Creates two .npy files in the specified directory:
        ///
        /// 1. `dataset.npy` - The vector data
        ///    - Shape: (num_vectors, N) where N is the vector dimensionality
        ///    - Data type: Same as the original dataset (e.g., i32, f32, etc.)
        ///    - Endianness: Native (little-endian on most modern systems)
        ///
        /// 2. `graph.npy` - The k-NN graph
        ///    - Shape: (num_nodes, degree) where degree is the graph degree
        ///    - Data type: UInt8 (if usize is 1 byte), UInt16 (2 bytes), UInt32 (4 bytes), or UInt64 (8 bytes)
        ///    - Endianness: Native (little-endian on most modern systems)
        ///
        /// The graph stores neighbor indices in row-major order:
        ///   graph[i * degree + j] = j-th neighbor of node i
        ///
        /// If the directory doesn't exist, it will be created.
        pub fn save(
            self: *const Self,
            dir_path: []const u8,
            allocator: std.mem.Allocator,
        ) !void {
            // Make sure the directory exists before trying to create the file
            std.fs.cwd().access(dir_path, .{}) catch |e| switch (e) {
                error.FileNotFound => try std.fs.cwd().makeDir(dir_path),
                else => return e,
            };

            var write_buffer: [4096]u8 = undefined;

            // Create the dataset file and writer
            const dataset_file_path = try std.fs.path.join(allocator, &[_][]const u8{ dir_path, DATASET_FILENAME });
            defer allocator.free(dataset_file_path);
            const dataset_file = try std.fs.cwd().createFile(dataset_file_path, .{});
            defer dataset_file.close();
            var dataset_file_writer = dataset_file.writer(&write_buffer);
            // Write the dataset to the file
            try self.dataset.toNpyFile(&dataset_file_writer.interface, allocator);
            try dataset_file_writer.interface.flush();

            // Create the graph file and writer
            const graph_file_path = try std.fs.path.join(allocator, &[_][]const u8{ dir_path, GRAPH_FILENAME });
            defer allocator.free(graph_file_path);
            const graph_file = try std.fs.cwd().createFile(graph_file_path, .{});
            defer graph_file.close();
            var graph_file_writer = graph_file.writer(&write_buffer);
            // Write the graph to the file
            try self.writeGraph(&graph_file_writer.interface, allocator);
            try graph_file_writer.interface.flush();
        }

        /// Writes the graph to a writer in .npy format.
        ///
        /// Data type: UInt8 (if usize is 1 byte), UInt16 (2 bytes), UInt32 (4 bytes), or UInt64 (8 bytes)
        /// based on the size of usize on the target platform.
        ///
        /// Endianness: Native (little-endian on most modern systems)
        ///
        /// The graph is stored as a 2D array with shape (num_nodes, num_neighbors_per_node).
        pub fn writeGraph(
            self: *const Self,
            writer: *std.io.Writer,
            allocator: std.mem.Allocator,
        ) !void {
            // Determine the appropriate element type based on usize size
            const element_type: znpy.ElementType = switch (@sizeOf(usize)) {
                1 => znpy.ElementType.UInt8,
                2 => znpy.ElementType{ .UInt16 = null },
                4 => znpy.ElementType{ .UInt32 = null },
                8 => znpy.ElementType{ .UInt64 = null },
                else => return error.UnsupportedUsizeSize,
            };

            // Write the graph in .npy format
            const header = znpy.header.Header{
                .descr = element_type,
                .order = .C,
                .shape = &[_]usize{ self.num_nodes, self.num_neighbors_per_node },
            };
            try header.writeAll(writer, allocator);
            try writer.writeAll(std.mem.sliceAsBytes(self.graph));
        }

        pub fn build(dataset: Dataset, config: BuildConfig, allocator: std.mem.Allocator) !Self {
            if (config.nn_descent_config.num_neighbors_per_node < config.graph_degree) {
                return error.InvalidBuildConfig;
            }

            log.info("Training initial graph with degree {d} using NN-Descent...", .{config.nn_descent_config.num_neighbors_per_node});
            var nn_descent = try NNDescent(T, N).init(
                dataset,
                config.nn_descent_config,
                allocator,
            );
            var timer = try std.time.Timer.start();
            nn_descent.train();
            nn_descent.sortNeighbors();
            const training_time = timer.read();
            log.info("NN Descent training time: {}ms\n", .{training_time / std.time.ns_per_ms});

            const neighbor_entries = nn_descent.neighbors_list.entries;

            // Free everything except the neighbor ids and the thread pool (which the optimizer will borrow)
            log.info("Freeing NN-Descent resources...", .{});
            allocator.free(neighbor_entries.items(.distance));
            allocator.free(neighbor_entries.items(.is_new));
            nn_descent.neighbor_candidates_new.deinit(allocator);
            nn_descent.neighbor_candidates_old.deinit(allocator);
            allocator.free(nn_descent.block_graph_updates_lists);
            allocator.free(nn_descent.block_graph_updates_buffer);
            allocator.free(nn_descent.graph_update_counts_buffer);
            allocator.free(nn_descent.node_ids_random);

            const num_nodes = nn_descent.neighbors_list.num_nodes;
            const num_neighbors_per_node = nn_descent.neighbors_list.num_neighbors_per_node;

            // We need to keep the neighbor IDs and the thread pool alive for the optimizer, so defer their cleanup
            const neighbor_ids: []usize = neighbor_entries.items(.neighbor_id);
            std.debug.assert(isValidGraph(neighbor_ids, num_nodes, num_neighbors_per_node));
            defer allocator.free(neighbor_ids);

            const thread_pool = nn_descent.thread_pool;
            defer if (thread_pool) |pool| {
                pool.deinit();
                allocator.destroy(pool);
            };

            const detour_counts: []usize = try allocator.alloc(usize, neighbor_entries.len);
            defer allocator.free(detour_counts);

            // Craft the optimizer entries by borrowing the neighbor IDs and detourable counts
            const optimizer_entries = mod_soa_slice.SoaSlice(NeighborsList(true).Entry){
                .ptrs = [_][*]u8{
                    @ptrCast(neighbor_ids.ptr),
                    @ptrCast(detour_counts.ptr),
                },
                .len = neighbor_entries.len,
            };

            // We let the optimizer borrow the entries and thread pool
            var optimizer = mod_optimizer.Optimizer.init(
                NeighborsList(true){
                    .entries = optimizer_entries,
                    .num_neighbors_per_node = num_neighbors_per_node,
                    .num_nodes = num_nodes,
                },
                thread_pool,
                nn_descent.num_nodes_per_block,
            );

            log.info("Optimizing graph with degree {d}...", .{config.graph_degree});
            timer.reset();
            const optimized_graph = try optimizer.optimize(config.graph_degree, allocator);
            const optimization_time = timer.read();
            log.info("Optimization time: {}ms\n", .{optimization_time / std.time.ns_per_ms});

            const graph_data: []const usize = optimized_graph.entries.items(.neighbor_id);
            std.debug.assert(isValidGraph(graph_data, num_nodes, config.graph_degree));

            return Self{
                .dataset = dataset,
                .graph = graph_data,
                .num_nodes = num_nodes,
                .num_neighbors_per_node = config.graph_degree,
            };
        }

        pub fn search(
            self: *const Self,
            queries: znpy.array.static.StaticArray(T, 2),
            config: mod_searcher.SearchConfig,
            allocator: std.mem.Allocator,
        ) !Searcher.SearchResult {
            const searcher = Searcher{
                .graph = self.graph,
                .dataset = self.dataset,
                .num_nodes = self.num_nodes,
                .num_neighbors_per_node = self.num_neighbors_per_node,
            };
            return searcher.search(&queries, &config, allocator);
        }
    };
}

test {
    _ = mod_nn_descent;
    _ = mod_optimizer;
    _ = mod_dataset;
    _ = mod_soa_slice;
    _ = mod_searcher;
}

// Integration test: build an index from a generated dataset, save it to disk, load it back,
// and assert the loaded index matches the original exactly.
fn testIndexRoundTrip(comptime T: type, comptime N: usize) !void {
    const Dataset = mod_dataset.Dataset(T, N);
    const IDX = Index(T, N);

    const num_vectors: usize = 50;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Create a small random dataset
    const dataset = try Dataset.initRandom(
        num_vectors,
        random,
        std.testing.allocator,
    );

    const graph_degree: usize = 4;
    const intermediate_degree: usize = 8; // must be >= graph_degree
    const block_size: usize = 16;
    const config = BuildConfig.init(
        graph_degree,
        intermediate_degree,
        num_vectors,
        1,
        42,
        block_size,
    );

    // Build the index (this consumes `dataset` into the returned index)
    var idx = try IDX.build(dataset, config, std.testing.allocator);
    defer idx.deinit(std.testing.allocator);

    // Directory to write the index to. save() will create it if necessary.
    const dir_path: []const u8 = "test_index_roundtrip";
    try idx.save(dir_path, std.testing.allocator);

    // Load the index back from disk
    var loaded = try IDX.load(dir_path, std.testing.allocator);
    defer loaded.deinit(std.testing.allocator);

    // Compare metadata
    try std.testing.expectEqual(idx.num_nodes, loaded.num_nodes);
    try std.testing.expectEqual(idx.num_neighbors_per_node, loaded.num_neighbors_per_node);
    try std.testing.expectEqual(idx.dataset.len, loaded.dataset.len);

    // Compare dataset contents and graph data exactly
    try std.testing.expectEqualSlices(T, idx.dataset.data_buffer, loaded.dataset.data_buffer);
    try std.testing.expectEqualSlices(usize, idx.graph, loaded.graph);

    // Cleanup files and directory created by the test to avoid leaving artifacts.
    // Remove dataset.npy and graph.npy then remove directory.
    const dataset_file_path = try std.fs.path.join(std.testing.allocator, &[_][]const u8{ dir_path, IDX.DATASET_FILENAME });
    defer std.testing.allocator.free(dataset_file_path);
    try std.fs.cwd().deleteFile(dataset_file_path);

    const graph_file_path = try std.fs.path.join(std.testing.allocator, &[_][]const u8{ dir_path, IDX.GRAPH_FILENAME });
    defer std.testing.allocator.free(graph_file_path);
    try std.fs.cwd().deleteFile(graph_file_path);

    // Remove the now-empty directory
    try std.fs.cwd().deleteDir(dir_path);
}

test "index round-trip" {
    try testIndexRoundTrip(i32, 128);
    try testIndexRoundTrip(i32, 256);
    try testIndexRoundTrip(i32, 512);
    try testIndexRoundTrip(f32, 128);
    try testIndexRoundTrip(f32, 256);
    try testIndexRoundTrip(f32, 512);
    try testIndexRoundTrip(f16, 128);
    try testIndexRoundTrip(f16, 256);
    try testIndexRoundTrip(f16, 512);
}
