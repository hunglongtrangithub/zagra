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
    ) Self {
        const nn_descent_config = TrainingConfig.init(
            intermediate_graph_degree,
            num_vectors,
            num_threads,
            seed,
        );
        return Self{
            .graph_degree = graph_degree,
            .nn_descent_config = nn_descent_config,
        };
    }
};

pub fn Index(comptime T: type, comptime N: usize) type {
    return struct {
        dataset: Dataset,
        graph: []usize,
        num_nodes: usize,
        num_neighbors_per_node: usize,

        const Dataset = mod_dataset.Dataset(T, N);
        const NeighborsList = mod_nn_descent.NeighborHeapList(T, true);

        const Self = @This();

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
            {
                // Free everything except the neighbors list and thread pool
                nn_descent.neighbor_candidates_new.deinit(allocator);
                nn_descent.neighbor_candidates_old.deinit(allocator);
                allocator.free(nn_descent.block_graph_updates_lists);
                allocator.free(nn_descent.block_graph_updates_buffer);
                allocator.free(nn_descent.graph_update_counts_buffer);
                allocator.free(nn_descent.node_ids_random);
            }

            // Extract the neighbor ID slice from the neighbors list and free everything else.
            // The neighbor entries should include only valid neighbor IDs with no empty slots.
            const neighbor_entries = nn_descent.neighbors_list.entries;
            const neighbor_ids: []usize = neighbor_entries.items(.neighbor_id);
            {
                allocator.free(neighbor_entries.items(.distance));
                allocator.free(neighbor_entries.items(.is_new));
            }

            const detour_counts: []usize = try allocator.alloc(usize, neighbor_entries.len);
            // Craft the optimizer entries by moving the neighbor IDs and detourable counts into the SoaSlice
            const optimizer_entries = mod_soa_slice.SoaSlice(mod_optimizer.Optimizer.Entry){
                .ptrs = [_][*]u8{
                    @ptrCast(neighbor_ids.ptr),
                    @ptrCast(detour_counts.ptr),
                },
                .len = neighbor_entries.len,
            };

            // We let the optimizer borrow the entries and thread pool
            const optimizer = try mod_optimizer.Optimizer.init(
                mod_optimizer.Optimizer.NeighborsList{
                    .entries = optimizer_entries,
                    .num_neighbors_per_node = nn_descent.neighbors_list.num_neighbors_per_node,
                    .num_nodes = nn_descent.neighbors_list.num_nodes,
                },
                nn_descent.thread_pool,
                nn_descent.num_nodes_per_block,
            );
            defer {
                optimizer_entries.deinit(allocator);
                if (nn_descent.thread_pool) |pool| {
                    pool.deinit();
                    allocator.destroy(pool);
                }
            }

            optimizer.optimize(config.graph_degree);
            // TODO: Extract the optimized neighbor IDs from the optimizer and construct the final graph structure for the index.
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
