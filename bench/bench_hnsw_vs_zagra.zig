const std = @import("std");
const config = @import("config");
const texmex = @import("texmex");
const znpy = @import("znpy");
const hnsw = @import("hnsw");
const zagra = @import("zagra");
const help = @import("help.zig");
const root = @import("root.zig");

const log = std.log.scoped(.ann_dataset);

const VectorSet = texmex.VectorSet;
const k = 10;

const BaseArray = znpy.array.static.StaticArray(f32, 2);
const QueryArray = znpy.array.static.StaticArray(f32, 2);
const GroundTruthArray = znpy.array.static.StaticArray(i32, 2);

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
    base_array: BaseArray,
    query_array: QueryArray,
    groundtruth: GroundTruthArray,
    name: []const u8,

    fn deinit(self: *AnnDataset, allocator: std.mem.Allocator) void {
        self.base_array.deinit(allocator);
        self.query_array.deinit(allocator);
        self.groundtruth.deinit(allocator);
        allocator.free(self.name);
    }

    fn numBase(self: *const AnnDataset) usize {
        return switch (self.base_array.shape.order) {
            .C => self.base_array.shape.dims[0],
            .F => self.base_array.shape.dims[1],
        };
    }

    fn numQuery(self: *const AnnDataset) usize {
        return switch (self.query_array.shape.order) {
            .C => self.query_array.shape.dims[0],
            .F => self.query_array.shape.dims[1],
        };
    }

    fn dimensions(self: *const AnnDataset) usize {
        return switch (self.base_array.shape.order) {
            .C => self.base_array.shape.dims[1],
            .F => self.base_array.shape.dims[0],
        };
    }

    fn gtK(self: *const AnnDataset) usize {
        return self.groundtruth.shape.dims[1];
    }

    fn load(
        allocator: std.mem.Allocator,
        dir_path: []const u8,
        name: []const u8,
    ) !AnnDataset {
        const dataset_dir_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_path, name });
        defer allocator.free(dataset_dir_path);

        log.info("Opening dataset directory: {s}", .{dataset_dir_path});

        var dir = if (std.fs.path.isAbsolute(dataset_dir_path))
            try std.fs.openDirAbsolute(dataset_dir_path, .{ .iterate = true })
        else
            try std.fs.cwd().openDir(dataset_dir_path, .{ .iterate = true });
        defer dir.close();

        var nullable_base_path: ?[]u8 = null;
        var nullable_query_path: ?[]u8 = null;
        var nullable_gt_path: ?[]u8 = null;
        defer {
            if (nullable_base_path) |p| allocator.free(p);
            if (nullable_query_path) |p| allocator.free(p);
            if (nullable_gt_path) |p| allocator.free(p);
        }

        var it = dir.iterate();
        while (try it.next()) |entry| {
            if (entry.kind != .file) continue;
            const entry_name = entry.name;

            if (std.mem.endsWith(u8, entry_name, "_base.npy")) {
                nullable_base_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dataset_dir_path, entry_name });
            } else if (std.mem.endsWith(u8, entry_name, "_query.npy")) {
                nullable_query_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dataset_dir_path, entry_name });
            } else if (std.mem.endsWith(u8, entry_name, "_groundtruth.npy")) {
                nullable_gt_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dataset_dir_path, entry_name });
            }
        }

        const base_path = nullable_base_path orelse {
            log.err("Could not find '*_base.npy' file", .{});
            return error.MissingBaseFile;
        };
        const query_path = nullable_query_path orelse {
            log.err("Could not find '*_query.npy' file", .{});
            return error.MissingQueryFile;
        };
        const gt_path = nullable_gt_path orelse {
            log.err("Could not find '*_groundtruth.npy' file", .{});
            return error.MissingGtFile;
        };

        log.info("Found files: base={s}, query={s}, gt={s}", .{ base_path, query_path, gt_path });

        const base_array = try loadNpyArray(f32, base_path, .@"64", allocator);
        const query_array = try loadNpyArray(f32, query_path, .@"64", allocator);
        const groundtruth = try loadNpyArray(i32, gt_path, .@"4", allocator);

        const dataset_name = try allocator.dupe(u8, name);

        log.info("Loaded: num_base={d}, num_query={d}, dimensions={d}", .{ switch (base_array.shape.order) {
            .C => base_array.shape.dims[0],
            .F => base_array.shape.dims[1],
        }, switch (query_array.shape.order) {
            .C => query_array.shape.dims[0],
            .F => query_array.shape.dims[1],
        }, switch (base_array.shape.order) {
            .C => base_array.shape.dims[1],
            .F => base_array.shape.dims[0],
        } });

        return AnnDataset{
            .base_array = base_array,
            .query_array = query_array,
            .groundtruth = groundtruth,
            .name = dataset_name,
        };
    }
};

fn loadNpyArray(comptime T: type, file_path: []const u8, comptime alignment: std.mem.Alignment, allocator: std.mem.Allocator) !znpy.array.static.StaticArray(T, 2) {
    const ArrayType = znpy.array.static.StaticArray(T, 2);

    const file = if (std.fs.path.isAbsolute(file_path))
        try std.fs.openFileAbsolute(file_path, .{})
    else
        try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var file_buffer: [4096]u8 = undefined;
    var file_reader = file.reader(&file_buffer);
    const reader = &file_reader.interface;

    return try ArrayType.fromFileAllocAligned(reader, alignment, allocator);
}

fn runHnswBenchmark(
    dataset: *const AnnDataset,
    allocator: std.mem.Allocator,
    cfg: HnswConfig,
    comptime k_val: usize,
) !HnswResult {
    const dim = dataset.dimensions();
    const num_base = dataset.numBase();
    const num_query = dataset.numQuery();
    const gt_k = dataset.gtK();

    log.info("Building HNSW index (dim={d}, M={d}, ef={d})...", .{ dim, cfg.M, cfg.ef_construction });

    var index = try hnsw.HierarchicalIndex.create(dim, num_base, cfg.M, cfg.ef_construction, 42, false);
    defer index.deinit();

    var timer = try std.time.Timer.start();
    for (0..num_base) |i| {
        const base_vec = dataset.base_array.data_buffer[i * dim ..][0..dim];
        try index.addPoint(base_vec, i, false);
    }
    const construction_ns = timer.read();
    log.info("HNSW construction: {d}ms", .{construction_ns / std.time.ns_per_ms});

    var labels = try allocator.alloc(usize, num_query * k_val);
    defer allocator.free(labels);
    var distances = try allocator.alloc(f32, num_query * k_val);
    defer allocator.free(distances);
    var query_times = try allocator.alloc(u64, num_query);
    defer allocator.free(query_times);

    log.info("Running HNSW search ({d} queries, k={d})...", .{ num_query, k_val });

    timer.reset();
    for (0..num_query) |qi| {
        const query_vec = dataset.query_array.data_buffer[qi * dim ..][0..dim];
        const labels_out = labels[qi * k_val ..][0..k_val];
        const distances_out = distances[qi * k_val ..][0..k_val];
        _ = try index.searchKnnWithEf(query_vec, k_val, cfg.ef_search, labels_out, distances_out);
        query_times[qi] = timer.lap();
    }

    var total_query_ns: u64 = 0;
    for (query_times) |t| total_query_ns += t;
    const avg_query_ns = total_query_ns / num_query;
    const throughput_qps = @as(f64, @floatFromInt(num_query)) * 1_000_000_000.0 / @as(f64, @floatFromInt(total_query_ns));

    log.info("HNSW: avg={d}ns, throughput={d:.1} qps", .{ avg_query_ns, throughput_qps });

    var recall_hits: usize = 0;
    for (0..num_query) |qi| {
        const gt_offset = qi * gt_k;
        const hnsw_offset = qi * k_val;
        for (0..k_val) |ki| {
            const label = labels[hnsw_offset + ki];
            for (0..gt_k) |gi| {
                if (label == @as(usize, @intCast(dataset.groundtruth.data_buffer[gt_offset + gi]))) {
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
    const dim = if (dataset.dimensions() != 128) {
        log.err("ZAGRA benchmark not supported for dim={d}. Supported: 128", .{dataset.dimensions()});
        return error.UnsupportedDimension;
    } else 128;
    const num_base = dataset.numBase();
    const num_query = dataset.numQuery();
    const gt_k = dataset.gtK();

    log.info("Building ZAGRA index (dim={d}, graph_degree={d})...", .{ dim, cfg.graph_degree });

    const ZagraIndex = zagra.Index(f32, dim);
    const ZagraDataset = zagra.Dataset(f32, dim);

    const base_data = @as([]align(64) const f32, @alignCast(dataset.base_array.data_buffer));
    const base_dataset = ZagraDataset{
        .data_buffer = base_data[0 .. num_base * dim],
        .len = num_base,
    };

    const build_config = zagra.index.BuildConfig.init(
        cfg.graph_degree,
        cfg.graph_degree * 2,
        num_base,
        null,
        42,
        16,
    );

    var timer = try std.time.Timer.start();
    var index = try ZagraIndex.build(base_dataset, build_config, allocator);
    defer index.deinit(allocator);
    const construction_ns = timer.read();
    log.info("ZAGRA construction: {d}ms", .{construction_ns / std.time.ns_per_ms});

    const query_data = @as([]align(64) const f32, @alignCast(dataset.query_array.data_buffer));
    const queries_shape = znpy.shape.StaticShape(2){
        .dims = [2]usize{ num_query, dim },
        .strides = [2]isize{ dim, 1 },
        .order = .C,
    };
    const queries = znpy.array.static.ConstStaticArray(f32, 2){
        .shape = queries_shape,
        .data_buffer = query_data[0 .. num_query * dim],
    };

    const search_config = zagra.index.SearchConfig{
        .k = k_val,
        .internal_k = cfg.internal_k,
        .max_iterations = cfg.max_iterations,
        .search_width = cfg.search_width,
        .num_threads = 1,
    };

    log.info("Running ZAGRA search ({d} queries, k={d})...", .{ num_query, k_val });

    timer.reset();
    var result = try index.search(queries, search_config, 42, allocator);
    defer {
        result.neighbors.deinit(allocator);
        result.distances.deinit(allocator);
    }
    const search_ns = timer.read();

    const avg_query_ns = search_ns / num_query;
    const throughput_qps = @as(f64, @floatFromInt(num_query)) * 1_000_000_000.0 / @as(f64, @floatFromInt(search_ns));

    log.info("ZAGRA: avg={d}ns, throughput={d:.1} qps", .{ avg_query_ns, throughput_qps });

    var recall_hits: usize = 0;
    for (0..num_query) |qi| {
        const gt_offset = qi * gt_k;
        for (0..k_val) |ki| {
            const label = @as(usize, @intCast(result.neighbors.data_buffer[qi * k_val + ki]));
            for (0..gt_k) |gi| {
                if (label == @as(usize, @intCast(dataset.groundtruth.data_buffer[gt_offset + gi]))) {
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
    inline for (std.meta.fieldNames(VectorSet)) |name| {
        try writer.print("  {s}\n", .{name});
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    const exe_path = args.next() orelse @src().file;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    const exe_name = std.fs.path.basename(exe_path[0..]);

    const vectorset_dir = args.next() orelse {
        try printUsage(stdout, exe_name);
        return;
    };
    try help.checkHelpWithUsage(stdout, vectorset_dir, exe_path, USAGE);

    const vectorset_name = args.next() orelse {
        try stdout.print("Error: Missing <vectorset_name>.\n", .{});
        try printUsage(stdout, exe_name);
        std.process.exit(1);
    };

    const result_prefix = args.next();

    const vector_set = std.meta.stringToEnum(VectorSet, vectorset_name) orelse {
        log.err("Unknown vectorset '{s}'", .{vectorset_name});
        std.process.exit(1);
    };
    if (vector_set == .ANN_SIFT1B) {
        log.err("ANN_SIFT1B is not yet supported", .{});
        std.process.exit(1);
    }

    var dataset = try AnnDataset.load(allocator, vectorset_dir, vectorset_name);
    defer dataset.deinit(allocator);

    const hnsw_cfg = HnswConfig{
        .M = 16,
        .ef_construction = 200,
        .ef_search = 100,
    };

    const zagra_cfg = ZagraConfig{
        .graph_degree = 32,
        .internal_k = 64,
        .search_width = 4,
        .max_iterations = 10,
    };

    const hnsw_result = try runHnswBenchmark(&dataset, allocator, hnsw_cfg, k);
    const zagra_result = try runZagraBenchmark(&dataset, allocator, zagra_cfg, k);

    const bench_result = BenchmarkResult{
        .dataset_name = dataset.name,
        .num_base = dataset.numBase(),
        .num_query = dataset.numQuery(),
        .dimensions = dataset.dimensions(),
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
