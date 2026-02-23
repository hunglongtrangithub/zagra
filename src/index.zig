const std = @import("std");
const builtin = @import("builtin");

const znpy = @import("znpy");

const mod_types = @import("types.zig");
const mod_dataset = @import("dataset.zig");
const mod_soa_slice = @import("index/soa_slice.zig");
const mod_optimizer = @import("index/optimizer.zig");
const mod_nn_descent = @import("index/nn_descent.zig");

pub const NNDescent = mod_nn_descent.NNDescent;
pub const TrainingConfig = mod_nn_descent.TrainingConfig;
pub const TrainingTiming = mod_nn_descent.TrainingTiming;
pub const SoaSlice = mod_soa_slice.SoaSlice;
pub const Optimizer = mod_optimizer.Optimizer;

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
        const nn_descent_config = TrainingConfig.init(
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

        const Dataset = mod_dataset.Dataset(T, N);
        const NeighborsList = mod_optimizer.Optimizer.NeighborsList;

        const Self = @This();

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.graph);
            self.dataset.deinit(allocator);
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
        ///    - Data type: u32 (if num_nodes < 2^32) or u64 (otherwise)
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
            const dataset_file_path = try std.fs.path.join(allocator, &[_][]const u8{ dir_path, "dataset.npy" });
            defer allocator.free(dataset_file_path);
            const dataset_file = try std.fs.cwd().createFile(dataset_file_path, .{});
            defer dataset_file.close();
            var dataset_file_writer = dataset_file.writer(&write_buffer);
            // Write the dataset to the file
            try self.dataset.toNpyFile(&dataset_file_writer.interface, allocator);
            try dataset_file_writer.interface.flush();

            // Create the graph file and writer
            const graph_file_path = try std.fs.path.join(allocator, &[_][]const u8{ dir_path, "graph.npy" });
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
        /// Data type:
        ///   - If num_nodes <= 2^32-1: uses u32 (UInt32)
        ///   - If num_nodes > 2^32-1: uses u64 (UInt64)
        ///
        /// Endianness: Native (little-endian on most modern systems)
        ///
        /// The graph is stored as a 2D array with shape (num_nodes, num_neighbors_per_node).
        pub fn writeGraph(
            self: *const Self,
            writer: *std.io.Writer,
            allocator: std.mem.Allocator,
        ) !void {
            // Determine the appropriate element type based on the number of nodes
            const element_type = if (self.num_nodes <= std.math.maxInt(u32))
                znpy.ElementType{ .UInt32 = null }
            else
                znpy.ElementType{ .UInt64 = null };

            // Write the graph in .npy format
            const header = znpy.header.Header{
                .descr = element_type,
                .order = znpy.Order.C,
                .shape = &[_]usize{ self.num_nodes, self.num_neighbors_per_node },
            };
            try header.writeAll(writer, allocator);
            try writer.writeAll(std.mem.sliceAsBytes(self.graph));
        }

        pub fn build(dataset: Dataset, config: BuildConfig, allocator: std.mem.Allocator) !Self {
            if (config.nn_descent_config.num_neighbors_per_node < config.graph_degree) {
                return error.InvalidBuildConfig;
            }

            var nn_descent = try NNDescent(T, N).init(
                dataset,
                config.nn_descent_config,
                allocator,
            );
            nn_descent.train();
            nn_descent.sortNeighbors();

            const neighbor_entries = nn_descent.neighbors_list.entries;

            // Free everything except the neighbor ids and the thread pool (which the optimizer will borrow)
            allocator.free(neighbor_entries.items(.distance));
            allocator.free(neighbor_entries.items(.is_new));
            nn_descent.neighbor_candidates_new.deinit(allocator);
            nn_descent.neighbor_candidates_old.deinit(allocator);
            allocator.free(nn_descent.block_graph_updates_lists);
            allocator.free(nn_descent.block_graph_updates_buffer);
            allocator.free(nn_descent.graph_update_counts_buffer);
            allocator.free(nn_descent.node_ids_random);

            // We need to keep the neighbor IDs and the thread pool alive for the optimizer, so defer their cleanup
            const neighbor_ids: []usize = neighbor_entries.items(.neighbor_id);
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
                    .num_neighbors_per_node = nn_descent.neighbors_list.num_neighbors_per_node,
                    .num_nodes = nn_descent.neighbors_list.num_nodes,
                },
                thread_pool,
                nn_descent.num_nodes_per_block,
            );

            const optimized_graph = try optimizer.optimize(config.graph_degree, allocator);

            return Self{
                .dataset = dataset,
                .graph = optimized_graph.entries.items(.neighbor_id),
                .num_nodes = nn_descent.neighbors_list.num_nodes,
                .num_neighbors_per_node = config.graph_degree,
            };
        }
    };
}

test {
    _ = mod_nn_descent;
    _ = mod_optimizer;
    _ = mod_dataset;
    _ = mod_soa_slice;
}

test "index" {
    _ = Index(f32, 128);
}
