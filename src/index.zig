const std = @import("std");

const mod_types = @import("types.zig");
const mod_dataset = @import("dataset.zig");
const mod_optimizer = @import("index/optimizer.zig");
pub const mod_nn_descent = @import("index/nn_descent.zig");

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
        const nn_descent_config = mod_nn_descent.TrainingConfig.init(
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
        graph: []isize,
        num_nodes: usize,
        num_neighbors_per_node: usize,

        const Dataset = mod_dataset.Dataset(T, N);
        const NNDescent = mod_nn_descent.NNDescent(T, N);
        const NeighborsList = mod_nn_descent.NeighborHeapList(T, true);
        const Optimizer = mod_optimizer.Optimizer;

        const Self = @This();

        pub fn build(dataset: Dataset, config: BuildConfig, allocator: std.mem.Allocator) !Self {
            var nn_descent = try NNDescent.init(
                dataset,
                config.nn_descent_config,
                allocator,
            );
            nn_descent.train();
            nn_descent.sortNeighbors();

            // const neighbor_entries = nn_descent.neighbors_list.entries;
            // const neighbor_ids: []isize = neighbor_entries.items(.neighbor_id);
            // const detourable_counts: []usize = try allocator.alloc(usize, neighbor_entries.len);
            // @memset(detourable_counts, 0);
            // const optimizer_entries = std.MultiArrayList(Optimizer.Entry){
            //     .len = neighbor_entries.len,
            //     .capacity = neighbor_entries.capacity,
            // };
        }
    };
}

test {
    _ = mod_nn_descent;
    _ = mod_optimizer;
    _ = mod_dataset;
}

test "index" {
    _ = Index(f32, 128);
}
