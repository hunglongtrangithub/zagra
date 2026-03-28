const std = @import("std");
const config = @import("config");
const texmex = @import("texmex");
const znpy = @import("znpy");
const hnsw = @import("hnsw");
const zagra = @import("zagra");
const help = @import("help.zig");
const root = @import("root.zig");

const log = std.log.scoped(.ann_dataset);

const k = 10;

const HnswConfig = struct {
    M: usize,
    ef_construction: usize,
    ef_search: usize,
};

const HnswResult = struct {
    construction_ns: u64,
    avg_query_ns: u64,
    throughput_qps: f64,
    recall: f64,
};

const ZagraConfig = struct {
    graph_degree: usize,
    intermediate_degree: usize,
    internal_k: usize,
    search_width: usize,
    max_iterations: usize,
};

const ZagraResult = struct {
    construction_ns: u64,
    avg_query_ns: u64,
    throughput_qps: f64,
    recall: f64,
};

const BenchmarkResult = struct {
    dataset_name: []const u8,
    num_base: usize,
    num_query: usize,
    dimensions: usize,
    k: usize,
    hnsw_config: HnswConfig,
    hnsw: HnswResult,
    zagra_config: ZagraConfig,
    zagra: ZagraResult,
    timestamp: i64,
};

const AnnDataset = struct {
    base_array: zagra.Dataset(f32, DIM),
    /// dims: `[num_query, 128]`, values are the query vectors
    query_array: znpy.array.StaticArray(f32, 2),
    /// dims: `[num_query, k]`, values are indices into the base set
    groundtruth: znpy.array.StaticArray(i32, 2),
    name: []const u8,

    /// Only 128-dimensional vectors are supported in this benchmark.
    pub const DIM = 128;

    const Error = error{
        InvalidArrayFile,
        OutOfMemory,
        ReadFailed,
        EndOfStream,
    };

    fn deinit(self: *AnnDataset, allocator: std.mem.Allocator) void {
        self.base_array.deinit(allocator);
        self.query_array.deinit(allocator);
        self.groundtruth.deinit(allocator);
        allocator.free(self.name);
    }

    /// Number of query vectors in the dataset.
    /// Both query_array and groundtruth both have the same number of queries.
    pub fn numQueries(self: *const AnnDataset) usize {
        return switch (self.query_array.shape.order) {
            .C => self.query_array.shape.dims[0],
            .F => self.query_array.shape.dims[1],
        };
    }

    /// Number of nearest neighbors (k) in the ground truth for each query.
    pub fn gtK(self: *const AnnDataset) usize {
        return switch (self.groundtruth.shape.order) {
            .C => self.groundtruth.shape.dims[1],
            .F => self.groundtruth.shape.dims[0],
        };
    }

    /// Directory for the vector set:
    /// <vectorset_dir_path>/<vectorset_name>/
    /// Inside the vector set directory, the following files are expected:
    /// - *_base.npy: base vectors (f32, shape [num_base, 128])
    /// - *_query.npy: query vectors (f32, shape [num_query, 128])
    /// - *_groundtruth.npy: ground truth neighbors (i32, shape [num_query, k])
    fn load(
        allocator: std.mem.Allocator,
        vectorset_dir_path: []const u8,
        vectorset_name: []const u8,
    ) !AnnDataset {
        // Get vector set directory
        var dir = if (std.fs.path.isAbsolute(vectorset_dir_path))
            try std.fs.openDirAbsolute(vectorset_dir_path, .{})
        else
            try std.fs.cwd().openDir(vectorset_dir_path, .{});
        defer dir.close();

        // Find sub-directory for the specified vector set
        var vectorset_dir = dir.openDir(
            vectorset_name,
            .{ .iterate = true },
        ) catch |e| switch (e) {
            error.FileNotFound => return error.VectorSetDirNotFound,
            else => return e,
        };
        defer vectorset_dir.close();

        var nullable_base_file: ?std.fs.File = null;
        var nullable_query_file: ?std.fs.File = null;
        var nullable_gt_file: ?std.fs.File = null;
        defer {
            if (nullable_base_file) |f| f.close();
            if (nullable_query_file) |f| f.close();
            if (nullable_gt_file) |f| f.close();
        }

        var it: std.fs.Dir.Iterator = vectorset_dir.iterate();
        while (try it.next()) |entry| {
            if (entry.kind != .file) continue;
            const entry_name = entry.name;

            if (std.mem.endsWith(u8, entry_name, "_base.npy")) {
                nullable_base_file = vectorset_dir.openFile(entry_name, .{}) catch return error.OpenBaseFileFailed;
            } else if (std.mem.endsWith(u8, entry_name, "_query.npy")) {
                nullable_query_file = vectorset_dir.openFile(entry_name, .{}) catch return error.OpenQueryFileFailed;
            } else if (std.mem.endsWith(u8, entry_name, "_groundtruth.npy")) {
                nullable_gt_file = vectorset_dir.openFile(entry_name, .{}) catch return error.OpenGroundTruthFileFailed;
            }
        }

        const base_file = nullable_base_file orelse return error.MissingBaseFile;
        const query_file = nullable_query_file orelse return error.MissingQueryFile;
        const gt_file = nullable_gt_file orelse return error.MissingGroundTruthFile;

        const base_array = loadZagraDataset(
            base_file,
            allocator,
        ) catch |e| return switch (e) {
            error.InvalidArrayFile => error.InvalidBaseArrayFile,
            else => e,
        };
        errdefer base_array.deinit(allocator);

        const query_array = loadNpyArray(
            f32,
            query_file,
            allocator,
        ) catch |e| return switch (e) {
            error.InvalidArrayFile => error.InvalidQueryArrayFile,
            else => e,
        };
        errdefer query_array.deinit(allocator);
        if (query_array.shape.dims[1] != DIM) {
            return error.UnsupportedDimension;
        }

        const groundtruth = loadNpyArray(
            i32,
            gt_file,
            allocator,
        ) catch |e| return switch (e) {
            error.InvalidArrayFile => error.InvalidGroundTruthArrayFile,
            else => e,
        };
        errdefer groundtruth.deinit(allocator);

        // Both groundtruth and query arrays should have the same number of queries
        const num_queries_qr = switch (query_array.shape.order) {
            .C => query_array.shape.dims[0],
            .F => query_array.shape.dims[1],
        };
        const num_queries_gt = switch (groundtruth.shape.order) {
            .C => groundtruth.shape.dims[0],
            .F => groundtruth.shape.dims[1],
        };
        if (num_queries_qr != num_queries_gt) {
            return error.InvalidGroundTruthArrayFile;
        }

        const dataset_name = try allocator.dupe(u8, vectorset_name);

        return AnnDataset{
            .base_array = base_array,
            .query_array = query_array,
            .groundtruth = groundtruth,
            .name = dataset_name,
        };
    }

    fn loadZagraDataset(
        file: std.fs.File,
        allocator: std.mem.Allocator,
    ) (error{UnsupportedDimension} || Error)!zagra.Dataset(f32, DIM) {
        var file_buffer: [4096]u8 = undefined;
        var file_reader = file.reader(&file_buffer);
        const reader = &file_reader.interface;

        return zagra.Dataset(f32, DIM).fromNpyFileReader(
            reader,
            allocator,
        ) catch |e| switch (e) {
            error.InvalidShape => error.UnsupportedDimension,
            error.OutOfMemory => Error.OutOfMemory,
            error.ReadFailed => Error.ReadFailed,
            error.EndOfStream => Error.EndOfStream,
            else => Error.InvalidArrayFile,
        };
    }

    fn loadNpyArray(
        comptime T: type,
        file: std.fs.File,
        allocator: std.mem.Allocator,
    ) Error!znpy.array.static.StaticArray(T, 2) {
        var file_buffer: [4096]u8 = undefined;
        var file_reader = file.reader(&file_buffer);
        const reader = &file_reader.interface;

        return znpy.array.StaticArray(T, 2).fromFileAlloc(
            reader,
            allocator,
        ) catch |e| switch (e) {
            error.OutOfMemory => Error.OutOfMemory,
            error.ReadFailed => Error.ReadFailed,
            error.EndOfStream => Error.EndOfStream,
            else => Error.InvalidArrayFile,
        };
    }
};

fn runHnswBenchmark(
    dataset: *const AnnDataset,
    allocator: std.mem.Allocator,
    cfg: HnswConfig,
    comptime k_val: usize,
) !HnswResult {
    if (cfg.ef_search < k_val) {
        return error.InvalidHnswConfig;
    }
    const dim = AnnDataset.DIM;
    const num_base = dataset.base_array.len;
    const num_query = dataset.numQueries();
    const gt_k = dataset.gtK();

    var timer = try std.time.Timer.start();
    var index = try hnsw.HierarchicalIndex.create(
        dim,
        num_base,
        cfg.M,
        cfg.ef_construction,
        42,
        false,
    );
    defer index.deinit();
    for (0..num_base) |i| {
        const base_vec = dataset.base_array.data_buffer[i * dim ..][0..dim];
        // label is the same as the index in the base array, which is what the ground truth uses
        try index.addPoint(base_vec, i, false);
    }
    const construction_ns = timer.read();

    var labels = try allocator.alloc(usize, num_query * k_val);
    defer allocator.free(labels);
    var distances = try allocator.alloc(f32, num_query * k_val);
    defer allocator.free(distances);

    timer.reset();
    for (0..num_query) |qi| {
        const query_vec = dataset.query_array.data_buffer[qi * dim ..][0..dim];
        const out_labels = labels[qi * k_val ..][0..k_val];
        const out_distances = distances[qi * k_val ..][0..k_val];
        const out_count = try index.searchKnnWithEf(
            query_vec,
            k_val,
            cfg.ef_search,
            out_labels,
            out_distances,
        );
        if (out_count != k_val) @panic("HNSW search returned fewer results than requested. This should not happen since ef_search >= k_val.");
    }

    const total_query_ns: u64 = timer.read();
    const avg_query_ns = total_query_ns / num_query;
    const throughput_qps = @as(f64, @floatFromInt(num_query)) / (@as(f64, @floatFromInt(total_query_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s)));

    log.info("HNSW: avg={d}ns, throughput={d:.1} qps", .{ avg_query_ns, throughput_qps });

    // Count recall hits from all queries
    var recall_hits: usize = 0;
    for (0..num_query) |qi| {
        const gt_offset = qi * gt_k;
        const hnsw_offset = qi * k_val;
        for (0..k_val) |ki| {
            const label = labels[hnsw_offset + ki];
            for (0..gt_k) |gi| {
                const gt_label = std.math.cast(usize, dataset.groundtruth.data_buffer[gt_offset + gi]) orelse
                    @panic("Ground truth label does not fit into usize");
                if (label == gt_label) {
                    recall_hits += 1;
                    break;
                }
            }
        }
    }
    const recall = @as(f64, @floatFromInt(recall_hits)) / @as(f64, @floatFromInt(num_query * k_val));
    log.info("HNSW recall@{}: {:.4}", .{ k_val, recall });

    return HnswResult{
        .construction_ns = construction_ns,
        .avg_query_ns = avg_query_ns,
        .throughput_qps = throughput_qps,
        .recall = recall,
    };
}

fn runZagraBenchmark(
    dataset: *const AnnDataset,
    allocator: std.mem.Allocator,
    cfg: ZagraConfig,
    comptime k_val: usize,
) !ZagraResult {
    const dim = AnnDataset.DIM;
    const num_base = dataset.base_array.len;
    const num_query = dataset.numQueries();
    const gt_k = dataset.gtK();

    log.info("Building ZAGRA index (dim={d}, graph_degree={d})...", .{ dim, cfg.graph_degree });

    const build_config = zagra.index.BuildConfig.init(
        cfg.graph_degree,
        cfg.intermediate_degree,
        num_base,
        null,
        42,
        16,
    );

    var timer = try std.time.Timer.start();
    var index = try zagra.Index(f32, dim).build(
        dataset.base_array,
        build_config,
        allocator,
    );
    defer index.deinit(allocator);
    const construction_ns = timer.read();
    log.info("ZAGRA construction: {d}ms", .{construction_ns / std.time.ns_per_ms});

    const search_config = zagra.index.SearchConfig{
        .k = k_val,
        .internal_k = cfg.internal_k,
        .max_iterations = cfg.max_iterations,
        .search_width = cfg.search_width,
        .num_threads = std.Thread.getCpuCount() catch 1,
    };

    log.info("Running ZAGRA search ({d} queries, k={d})...", .{ num_query, k_val });

    timer.reset();
    var result = try index.search(
        dataset.query_array.asConst(),
        search_config,
        42,
        allocator,
    );
    defer {
        result.neighbors.deinit(allocator);
        result.distances.deinit(allocator);
    }
    const search_ns = timer.read();

    const avg_query_ns = search_ns / num_query;
    const throughput_qps = @as(f64, @floatFromInt(num_query)) / (@as(f64, @floatFromInt(search_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s)));

    log.info("ZAGRA: avg={d}ns, throughput={d:.1} qps", .{ avg_query_ns, throughput_qps });

    var recall_hits: usize = 0;
    for (0..num_query) |qi| {
        const gt_offset = qi * gt_k;
        for (0..k_val) |ki| {
            const label = std.math.cast(usize, result.neighbors.data_buffer[qi * k_val + ki]) orelse
                @panic("ZAGRA returned neighbor label that does not fit into usize");
            for (0..gt_k) |gi| {
                const gt_label = std.math.cast(usize, dataset.groundtruth.data_buffer[gt_offset + gi]) orelse
                    @panic("Ground truth label does not fit into usize");
                if (label == gt_label) {
                    recall_hits += 1;
                    break;
                }
            }
        }
    }
    const recall = @as(f64, @floatFromInt(recall_hits)) / @as(f64, @floatFromInt(num_query * k_val));
    log.info("ZAGRA recall@{}: {:.4}", .{ k_val, recall });

    return ZagraResult{
        .construction_ns = construction_ns,
        .avg_query_ns = avg_query_ns,
        .throughput_qps = throughput_qps,
        .recall = recall,
    };
}

fn writeResults(result: *const BenchmarkResult, output_path: []const u8) !void {
    const output_file = try std.fs.cwd().createFile(output_path, .{});
    defer output_file.close();

    var file_buffer: [4096]u8 = undefined;
    var file_writer = output_file.writer(&file_buffer);

    var json_s = std.json.Stringify{
        .writer = &file_writer.interface,
        .options = .{ .whitespace = .indent_2 },
    };

    try json_s.beginObject();

    try json_s.objectField("dataset");
    try json_s.beginObject();
    try json_s.objectField("name");
    try json_s.write(result.dataset_name);
    try json_s.objectField("num_base");
    try json_s.write(result.num_base);
    try json_s.objectField("num_query");
    try json_s.write(result.num_query);
    try json_s.objectField("dimensions");
    try json_s.write(result.dimensions);
    try json_s.endObject();

    try json_s.objectField("k");
    try json_s.write(result.k);

    try json_s.objectField("hnsw");
    try json_s.beginObject();
    try json_s.objectField("config");
    try json_s.beginObject();
    try json_s.objectField("M");
    try json_s.write(result.hnsw_config.M);
    try json_s.objectField("ef_construction");
    try json_s.write(result.hnsw_config.ef_construction);
    try json_s.objectField("ef_search");
    try json_s.write(result.hnsw_config.ef_search);
    try json_s.endObject();
    try json_s.objectField("construction_time_ns");
    try json_s.write(result.hnsw.construction_ns);
    try json_s.objectField("avg_query_time_ns");
    try json_s.write(result.hnsw.avg_query_ns);
    try json_s.objectField("throughput_qps");
    try json_s.write(result.hnsw.throughput_qps);
    try json_s.objectField("recall");
    try json_s.write(result.hnsw.recall);
    try json_s.endObject();

    try json_s.objectField("zagra");
    try json_s.beginObject();
    try json_s.objectField("config");
    try json_s.beginObject();
    try json_s.objectField("graph_degree");
    try json_s.write(result.zagra_config.graph_degree);
    try json_s.objectField("internal_k");
    try json_s.write(result.zagra_config.internal_k);
    try json_s.objectField("search_width");
    try json_s.write(result.zagra_config.search_width);
    try json_s.objectField("max_iterations");
    try json_s.write(result.zagra_config.max_iterations);
    try json_s.endObject();
    try json_s.objectField("construction_time_ns");
    try json_s.write(result.zagra.construction_ns);
    try json_s.objectField("avg_query_time_ns");
    try json_s.write(result.zagra.avg_query_ns);
    try json_s.objectField("throughput_qps");
    try json_s.write(result.zagra.throughput_qps);
    try json_s.objectField("recall");
    try json_s.write(result.zagra.recall);
    try json_s.endObject();

    try json_s.objectField("timestamp");
    try json_s.write(result.timestamp);
    try json_s.endObject();

    try json_s.writer.flush();

    log.info("Results written to: {s}", .{output_path});
}

const USAGE =
    \\Usage: {s} <vectorset_dir> <vectorset_name> [result_prefix]
    \\Both absolute paths and relative paths to CWD are supported for <vectorset_dir>.
    \\
;

fn printUsage(writer: *std.io.Writer, exe_name: []const u8) !void {
    try writer.print(USAGE, .{exe_name});
    try writer.print("Available vectorsets:\n", .{});
    inline for (std.meta.fieldNames(texmex.VectorSet)) |name| {
        try writer.print("  {s}\n", .{name});
    }
}

pub fn main() !void {
    // var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
    // defer _ = gpa.deinit();
    // const allocator = gpa.allocator();

    const allocator = std.heap.page_allocator;

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    const exe_path = args.next() orelse @src().file;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    const exe_name = std.fs.path.basename(exe_path[0..]);

    const vectorset_dir = args.next() orelse {
        try stdout.print("Error: Missing <vectorset_dir>.\n", .{});
        try printUsage(stdout, exe_name);
        try stdout.flush();
        return;
    };
    try help.checkHelpWithUsage(
        stdout,
        vectorset_dir,
        exe_path,
        USAGE,
    );

    const vectorset_name = args.next() orelse {
        try stdout.print("Error: Missing <vectorset_name>.\n", .{});
        try printUsage(stdout, exe_name);
        try stdout.flush();
        std.process.exit(1);
    };

    const result_prefix = args.next();

    const vector_set = std.meta.stringToEnum(texmex.VectorSet, vectorset_name) orelse {
        log.err("Error: Unknown vectorset '{s}'", .{vectorset_name});
        std.process.exit(1);
    };
    if (vector_set == .ANN_SIFT1B) {
        log.err("ANN_SIFT1B is not yet supported", .{});
        std.process.exit(1);
    }

    var dataset = try AnnDataset.load(
        allocator,
        vectorset_dir,
        vectorset_name,
    );
    defer dataset.deinit(allocator);

    const hnsw_cfg = HnswConfig{
        .M = 16,
        .ef_construction = 200,
        .ef_search = 100,
    };

    const zagra_cfg = ZagraConfig{
        .graph_degree = 128,
        .intermediate_degree = 256,
        .internal_k = 10,
        .search_width = 10,
        .max_iterations = 100,
    };

    const hnsw_result = try runHnswBenchmark(
        &dataset,
        allocator,
        hnsw_cfg,
        k,
    );
    const zagra_result = try runZagraBenchmark(
        &dataset,
        allocator,
        zagra_cfg,
        k,
    );

    const bench_result = BenchmarkResult{
        .dataset_name = dataset.name,
        .num_base = dataset.base_array.len,
        .num_query = dataset.numQueries(),
        .dimensions = AnnDataset.DIM,
        .k = k,
        .hnsw_config = hnsw_cfg,
        .hnsw = hnsw_result,
        .zagra_config = zagra_cfg,
        .zagra = zagra_result,
        .timestamp = std.time.timestamp(),
    };

    const results_dir = root.RESULTS_DIR;
    std.fs.cwd().makePath(results_dir) catch |e| switch (e) {
        error.PathAlreadyExists => {},
        else => return e,
    };

    const output_path = if (result_prefix) |prefix|
        try std.fmt.allocPrint(allocator, results_dir ++ "/{s}_hnsw_vs_zagra.json", .{prefix})
    else
        results_dir ++ "/hnsw_vs_zagra.json";
    defer if (result_prefix) |_| allocator.free(output_path);
    try writeResults(&bench_result, output_path);
    try stdout.print("Benchmark completed successfully.\n", .{});
    try stdout.flush();

    // For some reason, the program hangs here. Force quit the process for now.
    std.process.exit(0);
}
