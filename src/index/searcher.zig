const std = @import("std");
const znpy = @import("znpy");
const log = std.log.scoped(.searcher);

const mod_dataset = @import("../dataset.zig");
const mod_vector = @import("../vector.zig");

/// Configuration for search in the index.
/// Reference: https://docs.rapids.ai/api/cuvs/stable/cpp_api/neighbors_cagra/#_CPPv4N4cuvs9neighbors5cagra13search_paramsE
pub const SearchConfig = struct {
    /// Number of nearest neighbors to return for each query.
    k: usize,
    /// Number of intermediate search results retained during the search.
    itopk_size: usize,
    max_iterations: usize,
    min_iterations: usize,
    search_width: usize,
    num_threads: usize,
};

/// Number of threads from an optional thread pool.
/// Returns 1 if thread_pool is null, otherwise returns the number of threads in the pool.
inline fn numThreads(thread_pool: ?*std.Thread.Pool) usize {
    return if (thread_pool) |pool| pool.threads.len else 1;
}

pub fn Searcher(comptime T: type, comptime N: usize) type {
    return struct {
        graph: []const usize,
        dataset: *const Dataset,
        num_nodes: usize,
        num_neighbors_per_node: usize,

        const ConstStaticArray = znpy.array.static.ConstStaticArray;
        const StaticArray = znpy.array.static.StaticArray;
        const Dataset = mod_dataset.Dataset(T, N);
        const Vector = mod_vector.Vector(T, N);

        pub const SearchResult = struct {
            neighbors: StaticArray(usize, 2),
            distances: StaticArray(T, 2),
        };

        pub const Error = error{
            InvalidSearchConfig,
            InvalidQueriesArray,
            SearchBufferSizeOverflow,
            NumThreadsTooLarge,
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
            allocator: std.mem.Allocator,
        ) (Error || std.mem.Allocator.Error)!SearchResult {
            if (config.itopk_size < config.k) return error.InvalidSearchConfig;
            if (config.itopk_size < config.search_width) return error.InvalidSearchConfig;

            const num_queries = verifyQueriesArray(queries) orelse return error.InvalidQueriesArray;
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
                config.itopk_size,
                std.math.mul(
                    usize,
                    config.search_width,
                    self.num_neighbors_per_node,
                ) catch return Error.SearchBufferSizeOverflow,
            ) catch return Error.SearchBufferSizeOverflow;
            const search_buffers = try allocator.alloc(usize, std.math.mul(
                usize,
                num_queries_per_block,
                search_buffer_size,
            ) catch return Error.NumThreadsTooLarge);
            defer allocator.free(search_buffers);

            const neighbors = StaticArray(usize, 2).init(
                [_]usize{ num_queries, config.k },
                .C,
                allocator,
            ) catch |e| return switch (e) {
                znpy.array.static.InitError.ShapeSizeOverflow => error.KTooLarge,
                else => e,
            };
            errdefer neighbors.deinit();

            const distances = StaticArray(T, 2).init(
                [_]usize{ num_queries, config.k },
                .C,
                allocator,
            ) catch |e| return switch (e) {
                znpy.array.static.InitError.ShapeSizeOverflow => error.KTooLarge,
                else => e,
            };
            errdefer distances.deinit();

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
                    queries,
                    thread_pool,
                    first_time_parent_flags_buffers,
                    first_time_candidate_flags_buffers,
                    search_buffers,
                    neighbors,
                    distances,
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
            queries: *const ConstStaticArray(T, 2),
            thread_pool: ?*std.Thread.Pool,
            first_time_parent_flags_buffers: []bool,
            first_time_candidate_flags_buffers: []bool,
            search_buffers: []usize,
            neighbors: *const StaticArray(usize, 2),
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

            const search_buffer_size = config.itopk_size + config.search_width * self.num_neighbors_per_node;
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
                            first_time_parent_flags_buffers[query_idx * num_nodes ..][0..num_nodes],
                            first_time_candidate_flags_buffers[query_idx * num_nodes ..][0..num_nodes],
                            search_buffers[query_idx * search_buffer_size ..][0..search_buffer_size],
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
                        first_time_parent_flags_buffers[query_idx * num_nodes ..][0..num_nodes],
                        first_time_candidate_flags_buffers[query_idx * num_nodes ..][0..num_nodes],
                        search_buffers[query_idx * search_buffer_size ..][0..search_buffer_size],
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
            first_time_parent_flags: []bool,
            first_time_candidate_flags: []bool,
            search_buffer: []usize,
            neighbors: []usize,
            distances: []T,
        ) void {
            const num_nodes = self.num_nodes;
            std.debug.assert(first_time_parent_flags.len == num_nodes);
            std.debug.assert(first_time_candidate_flags.len == num_nodes);

            const k = config.k;
            std.debug.assert(neighbors.len == k);
            std.debug.assert(distances.len == k);

            std.debug.assert(config.itopk_size >= config.k);
            std.debug.assert(config.itopk_size >= config.search_width);
            std.debug.assert(search_buffer.len == config.itopk_size + config.search_width * self.num_neighbors_per_node);

            std.debug.assert(std.mem.isAligned(@intFromPtr(query), 64));
            const query_vector_data: *const [N]T align(64) = @alignCast(query);
            const query_vector: *const Vector = @ptrCast(query_vector_data);

            _ = query_vector;
            @panic("searchThread is not implemented yet");
        }
    };
}

test Searcher {
    _ = Searcher(f32, 128);
}
