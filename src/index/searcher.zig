const std = @import("std");
const znpy = @import("znpy");
const log = std.log.scoped(.searcher);

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
    search_width: usize,
    num_threads: usize,
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
            neighbors: StaticArray(NodeIdType, 2),
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
            if (num_queries > std.math.maxInt(u64)) return Error.NumQueriesTooLarge;
            _ = std.math.add(u64, seed, @intCast(num_queries -| 1)) catch return Error.NumQueriesTooLarge;

            // Number of queries per block. 0 when number of queries or threads is 0.
            const num_queries_per_block = @min(config.num_threads, num_queries);

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
            // Fill the first internal_k entries of the search buffer with invalid nodes and infinite distances
            for (0..internal_k) |top_k_idx| {
                search_buffer.set(top_k_idx, .{
                    .node_id = num_nodes, // invalid node ID;
                    .distance = switch (elem_type) {
                        .Int32 => std.math.maxInt(T),
                        .Float, .Half => std.math.floatMax(T),
                    }, // Infinity distance
                });
            }

            // Fill the rest of the search buffer with random nodes and their distances to the query vector.
            for (0..search_buffer.len - internal_k) |candidate_idx| {
                const node_id = node_ids_random[(seed + candidate_idx) % self.num_nodes];
                const vector = self.dataset.getUnchecked(node_id);
                const distance = query_vector.sqdist(vector);
                search_buffer.set(internal_k + candidate_idx, .{
                    .node_id = node_id,
                    .distance = distance,
                });
            }

            const search_node_ids: []usize = search_buffer.items(.node_id);
            const search_distances: []T = search_buffer.items(.distance);
            const graph_degree = self.num_neighbors_per_node;
            var candidate_count = config.search_width;
            for (0..config.max_iterations) |iteration| {
                log.info("Iteration {d}, candidate_count: {d}", .{ iteration, candidate_count });
                const search_buffer_len = internal_k + candidate_count * graph_degree;

                // Fill the first internal_k entries with closest nodes in the search buffer.
                for (0..internal_k) |top_k_idx| {
                    const top_k_entry = search_buffer.get(top_k_idx);
                    // Find the entry with the smallest distance in the search buffer from top_k_idx to the end
                    var min_entry_idx = top_k_idx;
                    var min_entry = top_k_entry;
                    for (top_k_idx + 1..search_buffer_len) |entry_idx| {
                        const entry = search_buffer.get(entry_idx);
                        if (entry.distance < min_entry.distance) {
                            min_entry_idx = entry_idx;
                            min_entry = entry;
                        }
                    }
                    // Swap the entry with the smallest distance to index top_k_idx.
                    if (min_entry_idx != top_k_idx) {
                        search_buffer.set(top_k_idx, min_entry);
                        search_buffer.set(min_entry_idx, top_k_entry);
                    }
                }
                log.debug(
                    "Updated internal top k entries:\nNode IDs: {any}\nDistances: {any}",
                    .{ search_node_ids[0..internal_k], search_distances[0..internal_k] },
                );

                // Get search_width nodes from the top internal_k nodes that haven't been selected as parents before.
                // Skip ones that have alrady been parents.
                var new_candidate_count: usize = 0;
                for (0..config.search_width) |candidate_idx| {
                    const node_id: usize = search_node_ids[candidate_idx];
                    if (node_id == num_nodes) continue;
                    std.debug.assert(node_id < num_nodes);

                    if (first_time_parent_flags[node_id]) continue;
                    first_time_parent_flags[node_id] = true;

                    // Populate the search buffer with the neighbors of the selected node and their distances to the query vector.
                    const candidate_neighbors = self.graph[node_id * graph_degree ..][0..graph_degree];
                    for (candidate_neighbors, 0..) |neighbor_id, neighbor_idx| {
                        // TODO: Skip distance calculation with first_time_candidate_flags?
                        const vector = self.dataset.getUnchecked(neighbor_id);
                        const distance = query_vector.sqdist(vector);
                        const idx = internal_k + new_candidate_count * graph_degree + neighbor_idx;
                        search_node_ids[idx] = neighbor_id;
                        search_distances[idx] = distance;
                    }
                    new_candidate_count += 1;
                }
                log.debug("New candidate count: {}", .{new_candidate_count});

                if (new_candidate_count == 0) {
                    log.debug("No more new candidate counts found, stop searching.", .{});
                    break; // No new candidates, stop the search.
                }
                candidate_count = new_candidate_count;
            } else {
                log.debug("Search want through max {} iterations.", .{config.max_iterations});
            }

            // Fill the neighbors and distances arrays with the top k nodes in the search buffer.
            @memcpy(neighbors, search_node_ids[0..k]);
            @memcpy(distances, search_distances[0..k]);
        }
    };
}

test Searcher {
    _ = Searcher(f32, 128);
}
