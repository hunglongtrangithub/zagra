const std = @import("std");
const znpy = @import("znpy");
const log = std.log.scoped(.searcher);

const mod_index = @import("../index.zig");
const mod_types = @import("../types.zig");
const mod_dataset = @import("../dataset.zig");
const mod_vector = @import("../vector.zig");
const mod_soa_slice = @import("./soa_slice.zig");

const ConstStaticArray = znpy.array.static.ConstStaticArray;
const StaticArray = znpy.array.static.StaticArray;
const NodeIdType = mod_types.NodeIdType;

/// Configuration for search in the index.
/// Reference: https://docs.rapids.ai/api/cuvs/stable/cpp_api/neighbors_cagra/#_CPPv4N4cuvs9neighbors5cagra13search_paramsE
pub const SearchConfig = struct {
    /// Number of nearest neighbors to return for each query.
    k: usize,
    /// Number of intermediate search results retained during the search.
    internal_k: usize,
    max_iterations: usize,
    search_width: usize = 1,
    num_threads: usize = 1,

    /// Computes the maximum number of iterations for the search using CAGRA heuristic.
    pub fn maxIterations(
        internal_k: usize,
        search_width: usize,
        dataset_size: usize,
        graph_degree: usize,
    ) usize {
        // Graph expansion term
        const branching_factor: usize = @max(2, graph_degree / 2);

        // Base term: internal_k / search_width
        var max_iterations: usize = @max(1, internal_k / search_width);

        var num_reachable_nodes: usize = 1;
        while (num_reachable_nodes < dataset_size) {
            num_reachable_nodes *= branching_factor;
            max_iterations += 1;
        }

        return max_iterations;
    }
};

/// Number of threads from an optional thread pool.
/// Returns 1 if thread_pool is null, otherwise returns the number of threads in the pool.
inline fn numThreads(thread_pool: ?*std.Thread.Pool) usize {
    return if (thread_pool) |pool| pool.threads.len else 1;
}

pub fn Searcher(comptime T: type, comptime N: usize) type {
    const elem_type = mod_types.ElemType.fromZigType(T) orelse
        @compileError("Unsupported element type: " ++ @typeName(T));

    return struct {
        graph: []const usize,
        dataset: *const Dataset,
        num_nodes: usize,
        num_neighbors_per_node: usize,

        const Dataset = mod_dataset.Dataset(T, N);
        const Vector = mod_vector.Vector(T, N);

        const SearchEntry = struct {
            node_id: usize,
            distance: T,
        };
        const SearchBuffer = mod_soa_slice.SoaSlice(SearchEntry);

        pub const SearchResult = struct {
            /// dims: [num_queries, k]
            neighbors: StaticArray(NodeIdType, 2),
            /// dims: [num_queries, k]
            distances: StaticArray(T, 2),
        };

        pub const Error = error{
            InvalidSearchConfig,
            InvalidQueriesArray,
            SearchBufferSizeOverflow,
            NumThreadsTooLarge,
            NumQueriesTooLarge,
            KTooLarge,
        };

        const Self = @This();

        /// Verify that the queries array is valid for search:
        /// 1. The data buffer is 64-byte aligned for SIMD performance.
        /// 2. The shape is compatible with the expected number of dimensions (N) and number of queries.
        /// Returns the number of queries if valid, otherwise returns null.
        /// Accepts both C-order and F-order arrays:
        /// - C-order: dims[0] = num_queries, dims[1] = N
        /// - F-order: dims[0] = N, dims[1] = num_queries
        fn verifyQueriesArray(queries: *const ConstStaticArray(T, 2)) ?usize {
            if (!std.mem.isAligned(@intFromPtr(queries.data_buffer.ptr), 64)) {
                return null;
            }

            switch (queries.shape.order) {
                .C => {
                    if (queries.shape.dims[1] != N) {
                        return null;
                    }
                    return queries.shape.dims[0];
                },
                .F => {
                    if (queries.shape.dims[0] != N) {
                        return null;
                    }
                    return queries.shape.dims[1];
                },
            }
        }

        pub fn search(
            self: *const Self,
            queries: *const ConstStaticArray(T, 2),
            config: *const SearchConfig,
            seed: u64,
            allocator: std.mem.Allocator,
        ) (Error || std.mem.Allocator.Error)!SearchResult {
            if (config.internal_k < config.k) return Error.InvalidSearchConfig;
            if (config.internal_k < config.search_width) return Error.InvalidSearchConfig;

            const num_queries = verifyQueriesArray(queries) orelse return Error.InvalidQueriesArray;
            // Will seed each query with a different seed: from seed to seed + num_queries - 1. Guard against overflow of seed.
            _ = std.math.add(
                u64,
                seed,
                std.math.cast(u64, num_queries -| 1) orelse return Error.NumQueriesTooLarge,
            ) catch return Error.NumQueriesTooLarge;

            // Number of queries per block. 0 when number of queries or threads is 0.
            const num_queries_per_block: usize = @min(config.num_threads, num_queries);

            // Thread pool with num_queries_per_block threads if num_queries_per_block > 1, otherwise null.
            const thread_pool = if (num_queries_per_block != 1) blk: {
                const pool = try allocator.create(std.Thread.Pool);
                errdefer allocator.destroy(pool);
                pool.init(.{
                    .allocator = allocator,
                    .n_jobs = num_queries_per_block,
                }) catch return std.mem.Allocator.Error.OutOfMemory;
                break :blk pool;
            } else null;
            defer if (thread_pool) |pool| {
                pool.deinit();
                allocator.destroy(pool);
            };

            const first_time_parent_flags_buffers = try allocator.alloc(bool, std.math.mul(
                usize,
                num_queries_per_block,
                self.num_nodes,
            ) catch return Error.NumThreadsTooLarge);
            defer allocator.free(first_time_parent_flags_buffers);

            const first_time_candidate_flags_buffers = try allocator.alloc(bool, std.math.mul(
                usize,
                num_queries_per_block,
                self.num_nodes,
            ) catch return Error.NumThreadsTooLarge);
            defer allocator.free(first_time_candidate_flags_buffers);

            const search_buffer_size = std.math.add(
                usize,
                config.internal_k,
                std.math.mul(
                    usize,
                    config.search_width,
                    self.num_neighbors_per_node,
                ) catch return Error.SearchBufferSizeOverflow,
            ) catch return Error.SearchBufferSizeOverflow;
            var search_buffers = try SearchBuffer.init(
                std.math.mul(
                    usize,
                    num_queries_per_block,
                    search_buffer_size,
                ) catch return Error.NumThreadsTooLarge,
                allocator,
            );
            defer search_buffers.deinit(allocator);

            const neighbors = StaticArray(NodeIdType, 2).init(
                [_]usize{ num_queries, config.k },
                .C,
                allocator,
            ) catch |e| return switch (e) {
                znpy.array.static.InitError.ShapeSizeOverflow => Error.KTooLarge,
                else => return std.mem.Allocator.Error.OutOfMemory,
            };
            errdefer neighbors.deinit(allocator);

            const distances = StaticArray(T, 2).init(
                [_]usize{ num_queries, config.k },
                .C,
                allocator,
            ) catch |e| return switch (e) {
                znpy.array.static.InitError.ShapeSizeOverflow => Error.KTooLarge,
                else => return std.mem.Allocator.Error.OutOfMemory,
            };
            errdefer distances.deinit(allocator);

            const node_ids_random = try allocator.alloc(usize, self.num_nodes);
            defer allocator.free(node_ids_random);

            for (node_ids_random, 0..) |*node_id, i| node_id.* = i;
            var prng = std.Random.DefaultPrng.init(seed);
            const random = prng.random();
            random.shuffle(usize, node_ids_random);

            // If number of queries per block is 0, number of blocks will also be 0, so that no search will be performed.
            const num_blocks = std.math.divCeil(
                usize,
                num_queries,
                num_queries_per_block,
            ) catch 0;

            for (0..num_blocks) |block_id| {
                // Reset flags for each block to ensure correctness of search. This is necessary because flags are reused across blocks.
                @memset(first_time_parent_flags_buffers, false);
                @memset(first_time_candidate_flags_buffers, false);

                self.searchBlock(
                    block_id,
                    config,
                    seed,
                    queries,
                    thread_pool,
                    first_time_parent_flags_buffers,
                    first_time_candidate_flags_buffers,
                    node_ids_random,
                    search_buffers,
                    &neighbors,
                    &distances,
                );
            }

            return SearchResult{
                .neighbors = neighbors,
                .distances = distances,
            };
        }

        /// Perform search for a block of queries with a block ID.
        /// Block size is determined by the number of threads in the pool and the total number of queries.
        /// If block ID is out of range, no search will be performed.
        fn searchBlock(
            self: *const Self,
            block_id: usize,
            config: *const SearchConfig,
            seed: u64,
            queries: *const ConstStaticArray(T, 2),
            thread_pool: ?*std.Thread.Pool,
            first_time_parent_flags_buffers: []bool,
            first_time_candidate_flags_buffers: []bool,
            node_ids_random: []const usize,
            search_buffers: SearchBuffer,
            neighbors: *const StaticArray(NodeIdType, 2),
            distances: *const StaticArray(T, 2),
        ) void {
            std.debug.assert(verifyQueriesArray(queries) != null);

            const num_queries = queries.shape.dims[0];
            const k = config.k;
            std.debug.assert(neighbors.shape.dims[0] == num_queries and neighbors.shape.dims[1] == k and neighbors.shape.order == .C);
            std.debug.assert(distances.shape.dims[0] == num_queries and distances.shape.dims[1] == k and distances.shape.order == .C);

            const num_threads = numThreads(thread_pool);
            const num_nodes = self.num_nodes;
            std.debug.assert(first_time_parent_flags_buffers.len == num_threads * num_nodes);
            std.debug.assert(first_time_candidate_flags_buffers.len == num_threads * num_nodes);
            std.debug.assert(node_ids_random.len == self.num_nodes);

            const search_buffer_size = config.internal_k + config.search_width * self.num_neighbors_per_node;
            std.debug.assert(search_buffers.len == num_threads * search_buffer_size);

            // Query start is capped by num_queries to guard against out-of-range block_id.
            const query_start = @min(block_id * num_threads, num_queries);
            // num_queries_in_block <= num_threads.
            const num_queries_in_block = @min(query_start + num_threads, num_queries) - query_start;

            if (thread_pool) |pool| {
                var wait_group: std.Thread.WaitGroup = undefined;
                wait_group.reset();

                for (0..num_queries_in_block) |query_idx| {
                    const query_id = query_start + query_idx;
                    pool.spawnWg(
                        &wait_group,
                        searchThread,
                        .{
                            self,
                            queries.data_buffer[query_id * N ..][0..N],
                            config,
                            seed + @as(u64, @intCast(query_id)),
                            first_time_parent_flags_buffers[query_idx * num_nodes ..][0..num_nodes],
                            first_time_candidate_flags_buffers[query_idx * num_nodes ..][0..num_nodes],
                            node_ids_random,
                            search_buffers.subslice(query_idx * search_buffer_size, search_buffer_size),
                            neighbors.data_buffer[query_id * k ..][0..k],
                            distances.data_buffer[query_id * k ..][0..k],
                        },
                    );
                }
                pool.waitAndWork(&wait_group);
            } else {
                for (0..num_queries_in_block) |query_idx| {
                    const query_id = query_start + query_idx;
                    self.searchThread(
                        queries.data_buffer[query_id * N ..][0..N],
                        config,
                        seed + @as(u64, @intCast(query_id)),
                        first_time_parent_flags_buffers[query_idx * num_nodes ..][0..num_nodes],
                        first_time_candidate_flags_buffers[query_idx * num_nodes ..][0..num_nodes],
                        node_ids_random,
                        search_buffers.subslice(query_idx * search_buffer_size, search_buffer_size),
                        neighbors.data_buffer[query_id * k ..][0..k],
                        distances.data_buffer[query_id * k ..][0..k],
                    );
                }
            }
        }

        /// Initializes the search buffer for a new query search.
        ///
        /// The buffer is divided into two sections:
        /// - First `internal_k` entries: initialized with invalid node IDs (num_nodes) and maximum distances
        /// - Remaining entries: filled with random nodes from the dataset and their distances to the query
        fn initializeSearchBuffer(
            self: *const Self,
            query_vector: *const Vector,
            node_ids_random: []const usize,
            seed: u64,
            search_buffer: SearchBuffer,
            internal_k: usize,
        ) void {
            const num_nodes = self.num_nodes;
            std.debug.assert(search_buffer.len >= internal_k);

            for (0..internal_k) |top_k_idx| {
                search_buffer.set(top_k_idx, .{
                    .node_id = num_nodes,
                    .distance = switch (elem_type) {
                        .Int32 => std.math.maxInt(T),
                        .Float, .Half => std.math.floatMax(T),
                    },
                });
            }

            for (0..search_buffer.len - internal_k) |candidate_idx| {
                const node_id = node_ids_random[(seed + candidate_idx) % self.num_nodes];
                const vector = self.dataset.getUnchecked(node_id);
                const distance = query_vector.sqdist(vector);
                search_buffer.set(internal_k + candidate_idx, .{
                    .node_id = node_id,
                    .distance = distance,
                });
            }
        }

        /// Sorts the search buffer to find the top `internal_k` entries with smallest distances.
        ///
        /// Uses selection sort to place the `internal_k` closest nodes at the beginning of the buffer.
        /// Skips duplicate node IDs - if a node ID already exists in the sorted portion, it will
        /// not be added again, ensuring unique results.
        fn sortTopK(search_buffer: SearchBuffer, internal_k: usize) void {
            const node_ids: []const usize = search_buffer.items(.node_id);
            for (0..internal_k) |top_k_idx| {
                const top_k_entry = search_buffer.get(top_k_idx);
                var min_entry_idx = top_k_idx;
                var min_entry = top_k_entry;

                const existing_node_ids = node_ids[0..top_k_idx];
                for (top_k_idx + 1..search_buffer.len) |entry_idx| {
                    const entry = search_buffer.get(entry_idx);

                    // Prevent duplicate node IDs in the top k entries
                    if (std.mem.indexOfScalar(
                        usize,
                        existing_node_ids,
                        entry.node_id,
                    ) != null) continue;

                    if (entry.distance < min_entry.distance) {
                        min_entry_idx = entry_idx;
                        min_entry = entry;
                    }
                }
                if (min_entry_idx != top_k_idx) {
                    search_buffer.set(top_k_idx, min_entry);
                    search_buffer.set(min_entry_idx, top_k_entry);
                }
            }
        }

        /// Expands parent nodes to get new candidate neighbors for the next iteration.
        ///
        /// For each of the top `search_width` nodes in the internal top-k list that hasn't been
        /// expanded before, retrieves its neighbors from the graph and adds them to the candidate
        /// list portion of the search buffer.
        ///
        /// Uses `first_time_parent_flags` to track which nodes have been expanded as parents.
        /// Uses `first_time_candidate_flags` to track which nodes have been added as candidates.
        ///
        /// Returns the number of new candidates added in this iteration.
        fn updateSearchCandidates(
            self: *const Self,
            query_vector: *const Vector,
            search_buffer: SearchBuffer,
            first_time_parent_flags: []bool,
            first_time_candidate_flags: []bool,
            internal_k: usize,
            search_width: usize,
        ) usize {
            const num_nodes = self.num_nodes;
            std.debug.assert(first_time_parent_flags.len == num_nodes);
            std.debug.assert(first_time_candidate_flags.len == num_nodes);
            const graph_degree = self.num_neighbors_per_node;
            std.debug.assert(search_buffer.len >= internal_k + search_width * graph_degree);

            const search_node_ids: []usize = search_buffer.items(.node_id);
            const search_distances: []T = search_buffer.items(.distance);

            var new_candidate_count: usize = 0;
            for (0..search_width) |candidate_idx| {
                const node_id: usize = search_node_ids[candidate_idx];
                if (node_id >= num_nodes) continue;
                if (first_time_parent_flags[node_id]) continue;
                first_time_parent_flags[node_id] = true;

                const candidate_neighbors = self.graph[node_id * graph_degree ..][0..graph_degree];
                for (candidate_neighbors, 0..) |neighbor_id, neighbor_idx| {
                    if (first_time_candidate_flags[neighbor_id]) continue;
                    first_time_candidate_flags[neighbor_id] = true;

                    // TODO: Use first_time_candidate_flags to skip distance calculation.
                    const vector = self.dataset.getUnchecked(neighbor_id);
                    const distance = query_vector.sqdist(vector);
                    const idx = internal_k + new_candidate_count * graph_degree + neighbor_idx;
                    search_node_ids[idx] = neighbor_id;
                    search_distances[idx] = distance;
                }
                new_candidate_count += 1;
            }
            return new_candidate_count;
        }

        // Search for one query.
        pub fn searchThread(
            self: *const Self,
            query: *const [N]T,
            config: *const SearchConfig,
            seed: u64,
            first_time_parent_flags: []bool,
            first_time_candidate_flags: []bool,
            node_ids_random: []const usize,
            search_buffer: SearchBuffer,
            neighbors: []usize,
            distances: []T,
        ) void {
            const num_nodes = self.num_nodes;
            std.debug.assert(first_time_parent_flags.len == num_nodes);
            std.debug.assert(first_time_candidate_flags.len == num_nodes);
            std.debug.assert(node_ids_random.len == num_nodes);

            const k = config.k;
            std.debug.assert(neighbors.len == k);
            std.debug.assert(distances.len == k);

            std.debug.assert(config.internal_k >= config.k);
            std.debug.assert(config.internal_k >= config.search_width);
            std.debug.assert(search_buffer.len == config.internal_k + config.search_width * self.num_neighbors_per_node);

            std.debug.assert(std.mem.isAligned(@intFromPtr(query), 64));
            const query_vector_data: *const [N]T align(64) = @alignCast(query);
            const query_vector: *const Vector = @ptrCast(@alignCast(query_vector_data));

            const internal_k = config.internal_k;
            self.initializeSearchBuffer(
                query_vector,
                node_ids_random,
                seed,
                search_buffer,
                internal_k,
            );

            const search_node_ids: []usize = search_buffer.items(.node_id);
            const search_distances: []T = search_buffer.items(.distance);
            var candidate_count = config.search_width;
            for (0..config.max_iterations) |iteration| {
                log.debug("Iteration {d}, candidate_count: {d}", .{ iteration, candidate_count });

                sortTopK(search_buffer.subslice(
                    0,
                    internal_k + candidate_count * self.num_neighbors_per_node,
                ), internal_k);

                const new_candidate_count = self.updateSearchCandidates(
                    query_vector,
                    search_buffer,
                    first_time_parent_flags,
                    first_time_candidate_flags,
                    internal_k,
                    candidate_count,
                );
                std.debug.assert(new_candidate_count <= candidate_count);
                log.debug("New candidate count: {}", .{new_candidate_count});

                if (new_candidate_count == 0) {
                    log.debug("No more new candidate counts found, stop searching.", .{});
                    break;
                }
                candidate_count = new_candidate_count;
            } else {
                log.debug("Search went through max {} iterations.", .{config.max_iterations});
            }

            // Fill the neighbors and distances arrays with the top k nodes in the search buffer.
            @memcpy(neighbors, search_node_ids[0..k]);
            @memcpy(distances, search_distances[0..k]);
        }

        test "sortTopK sorts first internal_k entries" {
            var buffer = try SearchBuffer.init(10, std.testing.allocator);
            defer buffer.deinit(std.testing.allocator);

            buffer.set(0, .{ .node_id = 5, .distance = 5.0 });
            buffer.set(1, .{ .node_id = 3, .distance = 1.0 });
            buffer.set(2, .{ .node_id = 8, .distance = 8.0 });
            buffer.set(3, .{ .node_id = 1, .distance = 3.0 });
            buffer.set(4, .{ .node_id = 2, .distance = 2.0 });
            buffer.set(5, .{ .node_id = 9, .distance = 9.0 });
            buffer.set(6, .{ .node_id = 6, .distance = 6.0 });
            buffer.set(7, .{ .node_id = 4, .distance = 4.0 });
            buffer.set(8, .{ .node_id = 7, .distance = 7.0 });
            buffer.set(9, .{ .node_id = 0, .distance = 0.0 });

            const internal_k = 4;
            sortTopK(buffer, internal_k);

            const distances: []const f32 = buffer.items(.distance);
            const node_ids: []const usize = buffer.items(.node_id);

            try std.testing.expectEqual(0.0, distances[0]);
            try std.testing.expectEqual(1.0, distances[1]);
            try std.testing.expectEqual(2.0, distances[2]);
            try std.testing.expectEqual(3.0, distances[3]);

            try std.testing.expectEqual(0, node_ids[0]);
            try std.testing.expectEqual(3, node_ids[1]);
            try std.testing.expectEqual(2, node_ids[2]);
            try std.testing.expectEqual(1, node_ids[3]);

            for (0..internal_k) |i| {
                for (internal_k..buffer.len) |j| {
                    try std.testing.expect(distances[i] <= distances[j]);
                }
            }
        }

        test "sortTopK handles already sorted buffer" {
            var buffer = try SearchBuffer.init(5, std.testing.allocator);
            defer buffer.deinit(std.testing.allocator);

            buffer.set(0, .{ .node_id = 0, .distance = 1.0 });
            buffer.set(1, .{ .node_id = 1, .distance = 2.0 });
            buffer.set(2, .{ .node_id = 2, .distance = 3.0 });
            buffer.set(3, .{ .node_id = 3, .distance = 4.0 });
            buffer.set(4, .{ .node_id = 4, .distance = 5.0 });

            const internal_k = 3;
            sortTopK(buffer, internal_k);

            const distances = buffer.items(.distance);
            try std.testing.expectEqual(1.0, distances[0]);
            try std.testing.expectEqual(2.0, distances[1]);
            try std.testing.expectEqual(3.0, distances[2]);
        }

        test "sortTopK handles reverse sorted buffer" {
            var buffer = try SearchBuffer.init(5, std.testing.allocator);
            defer buffer.deinit(std.testing.allocator);

            buffer.set(0, .{ .node_id = 4, .distance = 5.0 });
            buffer.set(1, .{ .node_id = 3, .distance = 4.0 });
            buffer.set(2, .{ .node_id = 2, .distance = 3.0 });
            buffer.set(3, .{ .node_id = 1, .distance = 2.0 });
            buffer.set(4, .{ .node_id = 0, .distance = 1.0 });

            const internal_k = 3;
            sortTopK(buffer, internal_k);

            const distances = buffer.items(.distance);
            try std.testing.expectEqual(1.0, distances[0]);
            try std.testing.expectEqual(2.0, distances[1]);
            try std.testing.expectEqual(3.0, distances[2]);
        }

        test "sortTopK with internal_k equals buffer length" {
            var buffer = try SearchBuffer.init(4, std.testing.allocator);
            defer buffer.deinit(std.testing.allocator);

            buffer.set(0, .{ .node_id = 3, .distance = 9.0 });
            buffer.set(1, .{ .node_id = 1, .distance = 3.0 });
            buffer.set(2, .{ .node_id = 2, .distance = 5.0 });
            buffer.set(3, .{ .node_id = 0, .distance = 1.0 });

            const internal_k = 4;
            sortTopK(buffer, internal_k);

            const distances = buffer.items(.distance);
            try std.testing.expectEqual(1.0, distances[0]);
            try std.testing.expectEqual(3.0, distances[1]);
            try std.testing.expectEqual(5.0, distances[2]);
            try std.testing.expectEqual(9.0, distances[3]);
        }

        test "sortTopK prevents duplicate node IDs in top k" {
            var buffer = try SearchBuffer.init(8, std.testing.allocator);
            defer buffer.deinit(std.testing.allocator);

            buffer.set(0, .{ .node_id = 10, .distance = 5.0 });
            buffer.set(1, .{ .node_id = 20, .distance = 1.0 });
            buffer.set(2, .{ .node_id = 30, .distance = 3.0 });
            buffer.set(3, .{ .node_id = 10, .distance = 2.0 });
            buffer.set(4, .{ .node_id = 40, .distance = 4.0 });
            buffer.set(5, .{ .node_id = 50, .distance = 6.0 });
            buffer.set(6, .{ .node_id = 20, .distance = 7.0 });
            buffer.set(7, .{ .node_id = 60, .distance = 8.0 });

            const internal_k = 4;
            sortTopK(buffer, internal_k);

            const node_ids = buffer.items(.node_id);

            try std.testing.expectEqual(20, node_ids[0]);
            try std.testing.expectEqual(10, node_ids[1]);
            try std.testing.expectEqual(30, node_ids[2]);
            try std.testing.expectEqual(40, node_ids[3]);

            var seen = std.AutoArrayHashMap(usize, void).init(std.testing.allocator);
            defer seen.deinit();

            for (0..internal_k) |i| {
                try std.testing.expect(!seen.contains(node_ids[i]));
                try seen.put(node_ids[i], {});
            }
        }

        test "initializeSearchBuffer fills buffer correctly" {
            const TestN: usize = 128;
            const TestDataset = mod_dataset.Dataset(f32, TestN);

            const num_nodes: usize = 10;
            const graph_degree: usize = 2;
            const internal_k: usize = 4;
            const search_width: usize = 2;
            const search_buffer_size = internal_k + search_width * graph_degree;

            var prng = std.Random.DefaultPrng.init(42);
            const random = prng.random();
            var dataset = try TestDataset.initRandom(num_nodes, random, std.testing.allocator);
            defer dataset.deinit(std.testing.allocator);

            const graph = try std.testing.allocator.alloc(usize, num_nodes * graph_degree);
            defer std.testing.allocator.free(graph);
            for (0..num_nodes) |i| {
                graph[i * graph_degree] = (i + 1) % num_nodes;
                graph[i * graph_degree + 1] = (i + 2) % num_nodes;
            }

            const searcher = Self{
                .graph = graph,
                .dataset = &dataset,
                .num_nodes = num_nodes,
                .num_neighbors_per_node = graph_degree,
            };

            var buffer = try SearchBuffer.init(search_buffer_size, std.testing.allocator);
            defer buffer.deinit(std.testing.allocator);

            const node_ids_random = [_]usize{ 3, 7, 1, 9, 4, 2, 8, 0, 5, 6 };
            const seed: u64 = 42;

            const query_vector = dataset.getUnchecked(0);

            searcher.initializeSearchBuffer(
                query_vector,
                &node_ids_random,
                seed,
                buffer,
                internal_k,
            );

            const distances: []const f32 = buffer.items(.distance);
            const node_ids: []const usize = buffer.items(.node_id);

            for (0..internal_k) |i| {
                try std.testing.expectEqual(num_nodes, node_ids[i]);
                try std.testing.expectEqual(std.math.floatMax(f32), distances[i]);
            }

            for (internal_k..search_buffer_size) |i| {
                try std.testing.expect(node_ids[i] < num_nodes);
                try std.testing.expect(distances[i] >= 0.0);
            }
        }

        test "updateSearchCandidates expands parent nodes to neighbors" {
            const TestN: usize = 128;
            const TestDataset = mod_dataset.Dataset(f32, TestN);

            const num_nodes: usize = 4;
            const graph_degree: usize = 2;
            const internal_k: usize = 2;
            const search_width: usize = 2;

            var prng = std.Random.DefaultPrng.init(42);
            const random = prng.random();
            const dataset = try TestDataset.initRandom(num_nodes, random, std.testing.allocator);
            defer dataset.deinit(std.testing.allocator);

            const graph = [_]usize{
                1, 2, // neighbors of node 0
                0, 3, // neighbors of node 1
                0, 3, // neighbors of node 2
                1, 2, // neighbors of node 3
            };

            const searcher = Self{
                .graph = &graph,
                .dataset = &dataset,
                .num_nodes = num_nodes,
                .num_neighbors_per_node = graph_degree,
            };

            const search_buffer_size = internal_k + search_width * graph_degree;
            var search_buffer = try SearchBuffer.init(search_buffer_size, std.testing.allocator);
            defer search_buffer.deinit(std.testing.allocator);

            search_buffer.set(0, .{ .node_id = 0, .distance = 0.0 });
            search_buffer.set(1, .{ .node_id = 1, .distance = 1.0 });

            const first_time_parent_flags = try std.testing.allocator.alloc(bool, num_nodes);
            defer std.testing.allocator.free(first_time_parent_flags);
            @memset(first_time_parent_flags, false);

            const first_time_candidate_flags = try std.testing.allocator.alloc(bool, num_nodes);
            defer std.testing.allocator.free(first_time_candidate_flags);
            @memset(first_time_candidate_flags, false);

            const query_vector = dataset.getUnchecked(0);

            const new_count = searcher.updateSearchCandidates(
                query_vector,
                search_buffer,
                first_time_parent_flags,
                first_time_candidate_flags,
                internal_k,
                search_width,
            );

            try std.testing.expectEqual(2, new_count);

            const distances: []const f32 = search_buffer.items(.distance);
            const node_ids: []const usize = search_buffer.items(.node_id);

            try std.testing.expectEqual(1, node_ids[internal_k]);
            try std.testing.expectEqual(2, node_ids[internal_k + 1]);
            try std.testing.expectEqual(0, node_ids[internal_k + 2]);
            try std.testing.expectEqual(3, node_ids[internal_k + 3]);

            for (internal_k..internal_k + new_count * graph_degree) |i| {
                try std.testing.expect(distances[i] >= 0.0);
            }
        }
    };
}

test Searcher {
    _ = Searcher(f32, 128);
}
