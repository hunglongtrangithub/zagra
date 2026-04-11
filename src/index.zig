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

pub const NodeIdType = mod_types.NodeIdType;
pub const NNDescent = mod_nn_descent.NNDescent;
pub const NNDTrainingConfig = mod_nn_descent.TrainingConfig;
pub const NNDTrainingTiming = mod_nn_descent.TrainingTiming;
pub const NNDescentError = mod_nn_descent.NNDescentError;
pub const SoaSlice = mod_soa_slice.SoaSlice;
pub const Optimizer = mod_optimizer.Optimizer;
pub const SearchConfig = mod_searcher.SearchConfig;
pub const SearchError = mod_searcher.SearchError;

/// Check that the graph is valid:
/// - The length of neighbor_ids must equal number of nodes * number of neighbors per node
/// - All neighbor IDs must be in the range [0, num_nodes)
/// - No node can have itself as a neighbor
/// - No duplicate neighbor IDs for the same node
pub fn isValidGraph(
    neighbor_ids: []const usize,
    num_nodes: usize,
    num_neighbors_per_node: usize,
    allocator: std.mem.Allocator,
) bool {
    if (neighbor_ids.len != num_nodes * num_neighbors_per_node) {
        // Graph size does not match expected size
        return false;
    }

    var set: ?std.AutoArrayHashMapUnmanaged(usize, void) = .empty;
    set.?.ensureTotalCapacity(allocator, num_neighbors_per_node) catch {
        set = null;
    };
    defer if (set) |*s| s.deinit(allocator);

    for (0..num_nodes) |node_id| {
        defer if (set) |*s| s.clearRetainingCapacity();

        const start = node_id * num_neighbors_per_node;
        const end = start + num_neighbors_per_node;
        const slice = neighbor_ids[start..end];

        for (slice, 0..) |neighbor_id, neighbor_idx| {
            if (neighbor_id >= num_nodes) {
                // Invalid neighbor ID found
                return false;
            }

            if (neighbor_id == node_id) {
                // Node has itself as a neighbor
                return false;
            }

            // Find duplicate neighbors in the node's neighbors
            // Use the set for O(1) lookup if set is available, otherwise do O(n) scan
            const duplicate_found = blk: {
                if (set) |*s| {
                    if (s.contains(neighbor_id)) break :blk true else s.putAssumeCapacity(neighbor_id, {});
                } else for (slice[0..neighbor_idx]) |prev_neighbor_id| {
                    if (neighbor_id == prev_neighbor_id) break :blk true;
                }
                break :blk false;
            };

            if (duplicate_found) {
                // Duplicate neighbor found
                return false;
            }
        }
    }

    return true;
}

test "isValidGraph basic valid and invalid cases" {
    const allocator = std.testing.allocator;

    // Valid: 3 nodes, each with 2 neighbors, no self or duplicates, all in range
    const valid_neighbors = [_]usize{ 1, 2, 0, 2, 0, 1 };
    try std.testing.expect(isValidGraph(&valid_neighbors, 3, 2, allocator));

    // Invalid: wrong length
    const wrong_length = [_]usize{ 1, 2, 0, 2, 0 };
    try std.testing.expect(!isValidGraph(&wrong_length, 3, 2, allocator));

    // Invalid: neighbor out of range
    const out_of_range = [_]usize{ 1, 3, 0, 2, 0, 1 };
    try std.testing.expect(!isValidGraph(&out_of_range, 3, 2, allocator));

    // Invalid: self as neighbor
    const self_neighbor = [_]usize{ 0, 2, 0, 2, 0, 1 };
    try std.testing.expect(!isValidGraph(&self_neighbor, 3, 2, allocator));

    // Invalid: duplicate neighbor
    const duplicate_neighbor = [_]usize{ 1, 1, 0, 2, 0, 1 };
    try std.testing.expect(!isValidGraph(&duplicate_neighbor, 3, 2, allocator));
}

/// Configuration for building the index.
///
/// This includes parameters for both the initial graph construction (via NN-Descent)
/// and the subsequent graph optimization step.
///
/// The build process works as follows:
/// 1. NN-Descent constructs an initial k-NN graph using `nn_descent_config`
/// 2. The graph is pruned to the target `graph_degree`
pub const BuildConfig = struct {
    /// The target degree of the final k-NN graph (k).
    /// This is the number of nearest neighbors each node will have.
    graph_degree: usize,

    /// Configuration for the NN-Descent initial graph construction phase.
    /// Its `num_neighbors_per_node` must be >= `graph_degree`.
    nn_descent_config: mod_nn_descent.TrainingConfig,

    const Self = @This();

    pub fn init(
        graph_degree: usize,
        intermediate_graph_degree: usize,
        num_vectors: usize,
        num_threads: ?usize,
        seed: ?u64,
        block_size: ?usize,
    ) Self {
        return initExtended(
            graph_degree,
            intermediate_graph_degree,
            num_vectors,
            num_threads,
            seed,
            block_size,
            null,
            null,
        );
    }

    pub fn initExtended(
        graph_degree: usize,
        intermediate_graph_degree: usize,
        num_vectors: usize,
        num_threads: ?usize,
        seed: ?u64,
        block_size: ?usize,
        max_iterations: ?usize,
        max_candidates: ?usize,
    ) Self {
        var nn_descent_config = NNDTrainingConfig.init(
            intermediate_graph_degree,
            num_vectors,
            num_threads,
            seed,
        );
        if (max_iterations) |mi| nn_descent_config.max_iterations = mi;
        if (max_candidates) |mc| nn_descent_config.max_candidates = mc;
        if (block_size) |bs| nn_descent_config.block_size = bs;

        return Self{
            .graph_degree = graph_degree,
            .nn_descent_config = nn_descent_config,
        };
    }
};

pub const BuildError = error{InvalidBuildConfig} || NNDescentError || Optimizer.Error || std.mem.Allocator.Error;

/// Timing information for the index build process.
pub const BuildTiming = struct {
    /// NN-Descent init + train time in nanoseconds.
    nn_descent_ns: u64,
    /// Resource freeing time in nanoseconds.
    resource_free_ns: u64,
    /// Optimizer init + optimize time in nanoseconds.
    optimizer_ns: u64,
    /// Total build time in nanoseconds.
    total_ns: u64,
};

/// A generic ANN index combining a vector dataset with a k-NN graph.
///
/// Type parameters:
/// - `T`: Element type of vectors (e.g., f32, i32, f16). Must be in `ElemType`.
/// - `N`: Vector dimensionality. Must be in `DimType` (128, 256, or 512).
///
/// The index stores:
/// - A dataset of N-dimensional vectors with element type T
/// - A k-NN graph where each node has `num_neighbors_per_node` edges to nearest neighbors
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

        /// Frees all memory owned by this index.
        /// Does not free the dataset's backing memory if it was borrowed.
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.graph);
            self.dataset.deinit(allocator);
        }

        /// Loads the dataset from a .npy file reader.
        /// Validates that the shape has 2 dimensions and the second dimension equals N.
        fn loadDataset(
            reader: *std.Io.Reader,
            allocator: std.mem.Allocator,
        ) (error{InvalidDatasetDimension} || znpy.array.static.FromFileReaderError)!Dataset {
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
            reader: *std.Io.Reader,
            allocator: std.mem.Allocator,
            expected_num_nodes: usize,
        ) (error{
            InvalidGraphShape,
            GraphDatasetMismatch,
            InvalidGraphDtype,
            GraphDtypeTooLarge,
            InvalidGraph,
        } || znpy.header.ReadHeaderError || std.mem.Allocator.Error)!struct { []usize, usize } {
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

            if (!isValidGraph(
                graph_data,
                num_nodes,
                degree,
                allocator,
            )) return error.InvalidGraph;

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
            writer: *std.Io.Writer,
            allocator: std.mem.Allocator,
        ) (error{ UnsupportedUsizeSize, WriteFailed } || znpy.header.WriteHeaderError)!void {
            const element_type = znpy.ElementType.fromZigType(NodeIdType) catch {
                return error.UnsupportedUsizeSize;
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

        /// Internal build implementation for the ANN index.
        ///
        /// When `do_timing` is true, returns timing information.
        /// When `do_timing` is false, returns only the index.
        ///
        /// The build process:
        /// 1. NN-Descent constructs initial k-NN graph
        /// 2. Graph optimization prunes to target degree
        ///
        /// Arguments:
        /// - `dataset`: The vector dataset to build the index from (consumed)
        /// - `config`: Build configuration
        /// - `do_timing`: Compile-time flag to enable/disable timing
        /// - `allocator`: Memory allocator for building the index and timing data
        fn buildImpl(
            comptime do_timing: bool,
            dataset: Dataset,
            config: BuildConfig,
            allocator: std.mem.Allocator,
        ) (if (do_timing) (std.time.Timer.Error || BuildError) else BuildError)!struct {
            Self,
            if (do_timing) BuildTiming else void,
        } {
            if (config.nn_descent_config.num_neighbors_per_node < config.graph_degree) {
                return error.InvalidBuildConfig;
            }

            var timing: if (do_timing) BuildTiming else void = undefined;
            var timer_total = if (do_timing) try std.time.Timer.start() else {};
            var timer = if (do_timing) try std.time.Timer.start() else {};

            log.info("Training initial graph with degree {d} using NN-Descent...", .{config.nn_descent_config.num_neighbors_per_node});
            if (do_timing) timer.reset();
            var nn_descent = try NNDescent(T, N).init(
                &dataset,
                config.nn_descent_config,
                allocator,
            );
            nn_descent.train();
            nn_descent.sortNeighbors();
            if (do_timing) {
                timing.nn_descent_ns = timer.read();
                log.info("NN Descent time: {}ms\n", .{timing.nn_descent_ns / std.time.ns_per_ms});
            }

            const num_neighbor_entries = nn_descent.neighbors_list.entries.len;

            log.info("Freeing NN-Descent resources...", .{});
            if (do_timing) timer.reset();

            // Allocate optimizer entries first.
            // We copy neighbor IDs from nn_descent to optimizer_entries's neighbor_id slice.
            // optimizer_entries's detour_count slice will be initialized during optimization.
            var optimizer_entries = std.MultiArrayList(NeighborsList(true).Entry).empty;
            defer optimizer_entries.deinit(allocator);
            try optimizer_entries.ensureTotalCapacity(allocator, num_neighbor_entries);
            optimizer_entries.len = num_neighbor_entries;

            @memcpy(optimizer_entries.items(.neighbor_id), nn_descent.neighbors_list.entries.items(.neighbor_id));
            // Free everything in nn_descent except the thread pool. Optimizer will reuse the thread pool.
            nn_descent.neighbors_list.deinit(allocator);
            nn_descent.neighbor_candidates_new.deinit(allocator);
            nn_descent.neighbor_candidates_old.deinit(allocator);
            allocator.free(nn_descent.block_graph_updates_lists);
            allocator.free(nn_descent.block_graph_updates_buffer);
            allocator.free(nn_descent.graph_update_counts_buffer);
            allocator.free(nn_descent.node_ids_random);
            if (do_timing) {
                timing.resource_free_ns = timer.read();
                log.info("Resource free time: {}ms\n", .{timing.resource_free_ns / std.time.ns_per_ms});
            }

            const num_nodes = nn_descent.neighbors_list.num_nodes;
            const num_neighbors_per_node = nn_descent.neighbors_list.num_neighbors_per_node;

            std.debug.assert(isValidGraph(
                optimizer_entries.items(.neighbor_id),
                num_nodes,
                num_neighbors_per_node,
                allocator,
            ));

            const thread_pool = nn_descent.thread_pool;
            defer if (thread_pool) |pool| {
                pool.deinit();
                allocator.destroy(pool);
            };

            log.info("Optimizing graph with degree {d}...", .{config.graph_degree});
            if (do_timing) timer.reset();
            var optimizer = mod_optimizer.Optimizer.init(
                NeighborsList(true){
                    .entries = optimizer_entries.slice(),
                    .num_neighbors_per_node = num_neighbors_per_node,
                    .num_nodes = num_nodes,
                },
                thread_pool,
                nn_descent.num_nodes_per_block,
            );
            var optimized_graph = try optimizer.optimize(config.graph_degree, allocator);
            defer optimized_graph.deinit(allocator);
            if (do_timing) {
                timing.optimizer_ns = timer.read();
                log.info("Optimizer time: {}ms\n", .{timing.optimizer_ns / std.time.ns_per_ms});
            }

            const graph_data: []const usize = try allocator.dupe(usize, optimized_graph.entries.items(.neighbor_id));
            std.debug.assert(isValidGraph(
                graph_data,
                num_nodes,
                config.graph_degree,
                allocator,
            ));

            if (do_timing) timing.total_ns = timer_total.read();
            return .{
                Self{
                    .dataset = dataset,
                    .graph = graph_data,
                    .num_nodes = num_nodes,
                    .num_neighbors_per_node = config.graph_degree,
                },
                timing,
            };
        }

        /// Builds an index from a dataset.
        ///
        /// The build process consists of two phases:
        /// 1. NN-Descent constructs an initial k-NN graph using the provided config
        /// 2. Graph optimization prunes and refines the graph to the target degree
        pub fn build(
            dataset: Dataset,
            config: BuildConfig,
            allocator: std.mem.Allocator,
        ) BuildError!Self {
            const index, _ = try buildImpl(
                false,
                dataset,
                config,
                allocator,
            );
            return index;
        }

        /// Builds an index from a dataset, logging timing information.
        ///
        /// Same as `build`, but also returns timing information for each phase.
        pub fn buildWithTiming(
            dataset: Dataset,
            config: BuildConfig,
            allocator: std.mem.Allocator,
        ) (std.time.Timer.Error || BuildError)!struct { Self, BuildTiming } {
            return buildImpl(
                true,
                dataset,
                config,
                allocator,
            );
        }

        /// Searches the index for the k nearest neighbors of the given query vectors.
        ///
        /// Arguments:
        /// - `queries`: A 2D array of query vectors (shape: (num_queries, N))
        /// - `config`: Search configuration (ef construction, k for k-NN, etc.)
        /// - `seed`: Optional random seed for tie-breaking. If null, a random seed is used.
        /// - `allocator`: Memory allocator for the result
        ///
        /// Returns the search results containing distances and indices of nearest neighbors.
        pub fn search(
            self: *const Self,
            queries: znpy.array.static.ConstStaticArray(T, 2),
            config: mod_searcher.SearchConfig,
            seed: ?u64,
            allocator: std.mem.Allocator,
        ) (SearchError || std.mem.Allocator.Error)!Searcher.SearchResult {
            const searcher = Searcher{
                .graph = self.graph,
                .dataset = &self.dataset,
                .num_nodes = self.num_nodes,
                .num_neighbors_per_node = self.num_neighbors_per_node,
            };
            return searcher.search(
                &queries,
                &config,
                if (seed) |s| s else std.crypto.random.int(u64),
                allocator,
            );
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

/// Integration test for index save/load round-trip.
///
/// Builds an index from a generated dataset, saves it to disk, loads it back,
/// and asserts the loaded index matches the original exactly.
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

test "Index - build() and buildWithTiming() produce identical results" {
    const T = f32;
    const N = 128;
    const Dataset = mod_dataset.Dataset(T, N);
    const Idx = Index(T, N);

    const num_vectors: usize = 50;

    const graph_degree: usize = 4;
    const intermediate_degree: usize = 8;
    const block_size: usize = 16;
    const config = BuildConfig.init(
        graph_degree,
        intermediate_degree,
        num_vectors,
        1,
        42,
        block_size,
    );

    var prng = std.Random.DefaultPrng.init(42);
    const dataset = try Dataset.initRandom(
        num_vectors,
        prng.random(),
        std.testing.allocator,
    );
    defer dataset.deinit(std.testing.allocator);

    const idx_build = try Idx.build(
        dataset,
        config,
        std.testing.allocator,
    );
    defer std.testing.allocator.free(idx_build.graph);

    const idx_timing, _ = try Idx.buildWithTiming(
        dataset,
        config,
        std.testing.allocator,
    );
    defer std.testing.allocator.free(idx_timing.graph);

    try std.testing.expectEqualSlices(T, idx_build.dataset.data_buffer, idx_timing.dataset.data_buffer);
    try std.testing.expectEqual(idx_build.num_nodes, idx_timing.num_nodes);
    try std.testing.expectEqual(idx_build.num_neighbors_per_node, idx_timing.num_neighbors_per_node);
    try std.testing.expectEqualSlices(usize, idx_build.graph, idx_timing.graph);
}
