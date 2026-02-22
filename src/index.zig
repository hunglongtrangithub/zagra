const std = @import("std");

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
        dataset: Dataset,
        graph: []const usize,
        num_nodes: usize,
        num_neighbors_per_node: usize,

        const Dataset = mod_dataset.Dataset(T, N);
        const NeighborsList = mod_optimizer.Optimizer.NeighborsList;

        const Self = @This();

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.graph);
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
