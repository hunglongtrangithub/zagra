const std = @import("std");
const config = @import("config");

var stdin_buffer: [1024]u8 = undefined;
var stdin_reader = std.fs.File.stdin().reader(&stdin_buffer);
const stdin = &stdin_reader.interface;

var stdout_buffer: [1024]u8 = undefined;
var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
const stdout = &stdout_writer.interface;

var stderr_buffer: [1024]u8 = undefined;
var stderr_writer = std.fs.File.stderr().writer(&stderr_buffer);
const stderr = &stderr_writer.interface;

const URL_PREFIX = "ftp://ftp.irisa.fr/local/texmex/corpus/";

const VectorSet = enum {
    ANN_SIFT10K,
    ANN_SIFT1M,
    ANN_GIST1M,
    ANN_SIFT1B,

    const SpawnResult = struct {
        child: std.process.Child,
        stdout: []u8,
        stderr: []u8,

        pub fn deinit(self: *const SpawnResult, allocator: std.mem.Allocator) void {
            allocator.free(self.stdout);
            allocator.free(self.stderr);
        }
    };

    fn checkExecutable(name: []const u8, allocator: std.mem.Allocator) std.io.Writer.Error!void {
        const run_result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{ "which", name },
        }) catch |e| {
            try stderr.print("Error starting 'which {s}': {}\n", .{ name, e });
            try stderr.flush();
            std.process.exit(1);
        };
        allocator.free(run_result.stdout);
        allocator.free(run_result.stderr);
        if (run_result.term.Exited != 0) {
            try stderr.print("Error: '{s}' is not installed or not in PATH.\n", .{name});
            try stderr.flush();
            std.process.exit(1);
        }
    }

    fn spawnAndCollectOutput(argv: []const []const u8, allocator: std.mem.Allocator) !SpawnResult {
        var child = std.process.Child.init(argv, allocator);
        child.stdin_behavior = .Ignore;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;

        var stdout_output: std.ArrayList(u8) = .empty;
        defer stdout_output.deinit(allocator);
        var stderr_output: std.ArrayList(u8) = .empty;
        defer stderr_output.deinit(allocator);

        try child.spawn();
        errdefer {
            _ = child.kill() catch {};
        }

        child.collectOutput(allocator, &stdout_output, &stderr_output, 50 * 1024) catch |e| {
            std.log.err("Error collecting output: {}", .{e});
        };

        return .{
            .child = child,
            .stdout = try stdout_output.toOwnedSlice(allocator),
            .stderr = try stderr_output.toOwnedSlice(allocator),
        };
    }

    /// Spawns a child process with the given arguments and waits for it to finish.
    /// Exits on spawn failure or non-zero exit code.
    fn spawnAndWait(
        argv: []const []const u8,
        allocator: std.mem.Allocator,
    ) std.io.Writer.Error!void {
        if (argv.len == 0) return;
        const exe = argv[0];
        var cmd = std.process.Child.init(argv, allocator);
        const term = cmd.spawnAndWait() catch |e| {
            try stderr.print("Error starting {s}: {}\n", .{ exe, e });
            try stderr.flush();
            std.process.exit(1);
        };
        if (term.Exited != 0) {
            try stderr.print("Error: {s} failed with exit code {d}.\n", .{ exe, term.Exited });
            try stderr.flush();
            std.process.exit(1);
        }
    }

    /// Make dataset directory if it doesn't exist, or verify access if it does. Exits on failure.
    /// `dataset_dir` should be a relative path under the current working directory.
    fn makeDatasetDir(dataset_dir: []const u8) std.io.Writer.Error!std.fs.Dir {
        std.log.info("Checking dataset directory: {s}", .{dataset_dir});
        if (std.fs.cwd().access(dataset_dir, .{})) {
            std.log.info("Dataset directory already exists: {s}", .{dataset_dir});
        } else |err| {
            switch (err) {
                error.FileNotFound => {
                    std.log.info("Creating dataset directory: {s}", .{dataset_dir});

                    std.fs.cwd().makeDir(dataset_dir) catch |e| {
                        try stderr.print("Error creating dataset directory: {}\n", .{e});
                        try stderr.flush();
                        std.process.exit(1);
                    };
                },
                else => {
                    try stderr.print("Error accessing dataset directory: {}\n", .{err});
                    try stderr.flush();
                    std.process.exit(1);
                },
            }
        }

        // Try opening the directory
        return std.fs.cwd().openDir(dataset_dir, .{ .iterate = true }) catch |e| {
            try stderr.print("Error opening dataset directory: {}\n", .{e});
            try stderr.flush();
            std.process.exit(1);
        };
    }

    pub fn install(self: VectorSet, allocator: std.mem.Allocator) (std.mem.Allocator.Error || std.io.Writer.Error)!void {
        // Check if executables are available
        const executables = [_][]const u8{ "wget", "tar", "gzip" };
        for (executables) |exe| {
            try checkExecutable(exe, allocator);
        }

        const dataset_dir_str = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ config.DATASET_DIR, @tagName(self) });
        defer allocator.free(dataset_dir_str);
        const dataset_dir = try makeDatasetDir(dataset_dir_str);

        const valid_extensions = [_][]const u8{ ".fvecs", ".ivecs", ".bvecs" };

        switch (self) {
            .ANN_SIFT10K, .ANN_SIFT1M, .ANN_GIST1M => {
                const tar_file_path = try std.fmt.allocPrint(allocator, "{s}/{s}.tar.gz", .{ dataset_dir_str, @tagName(self) });
                defer allocator.free(tar_file_path);
                const url = try std.fmt.allocPrint(allocator, "{s}{s}", .{ URL_PREFIX, switch (self) {
                    .ANN_SIFT10K => "siftsmall.tar.gz",
                    .ANN_SIFT1M => "sift.tar.gz",
                    .ANN_GIST1M => "gist.tar.gz",
                    else => unreachable,
                } });
                defer allocator.free(url);

                try spawnAndWait(&[_][]const u8{
                    "wget",
                    "-O",
                    tar_file_path,
                    url,
                }, allocator);
                try spawnAndWait(&[_][]const u8{
                    "tar",
                    "-xzf",
                    tar_file_path,
                    "-C",
                    dataset_dir_str,
                }, allocator);
            },
            .ANN_SIFT1B => {
                // ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
                // ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs.gz
                // ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
                // ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
                const sets = [_][]const u8{
                    "bigann_base.bvecs.gz",
                    "bigann_learn.bvecs.gz",
                    "bigann_query.bvecs.gz",
                    "bigann_gnd.tar.gz",
                };

                const file_paths = try allocator.alloc([]const u8, sets.len);
                for (sets, 0..) |set, i| {
                    file_paths[i] = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dataset_dir_str, set });
                }
                defer {
                    for (file_paths) |path| {
                        allocator.free(path);
                    }
                    allocator.free(file_paths);
                }

                const spawn_results = try allocator.alloc(SpawnResult, sets.len);
                defer {
                    for (spawn_results) |result| {
                        result.deinit(allocator);
                    }
                    allocator.free(spawn_results);
                }

                // Download each set with wget in parallel
                inline for (sets, file_paths, 0..) |set, file_path, i| {
                    const spawn_result = spawnAndCollectOutput(&[_][]const u8{
                        "wget",
                        "-O",
                        file_path,
                        URL_PREFIX ++ set,
                    }, allocator) catch |e| switch (e) {
                        error.OutOfMemory => return std.mem.Allocator.Error.OutOfMemory,
                        else => {
                            try stderr.print("Error starting wget for {s}: {}\n", .{ set, e });
                            try stderr.flush();
                            std.process.exit(1);
                        },
                    };
                    spawn_results[i] = spawn_result;
                }

                // Wait and check exit code. If non-zero, print error and exit. If zero, continue to extract/move files.
                for (spawn_results, sets) |*result, set| {
                    const term = result.child.wait() catch |e| {
                        try stderr.print("[{s}] Error waiting for {s} to finish: {}\n", .{ set, result.child.argv[0], e });
                        try stderr.flush();
                        std.process.exit(1);
                    };
                    if (term.Exited != 0) {
                        try stderr.print("[{s}] Error: {s} failed with exit code {d}. stderr:\n{s}\n", .{ set, result.child.argv[0], term.Exited, result.stderr });
                        try stderr.flush();
                        std.process.exit(1);
                    } else {
                        std.log.info("{s} downloaded successfully", .{set});
                    }
                }

                // Extract the ground truth set with tar, and all other sets with gzip, sequentially
                for (sets, file_paths) |set, file_path| {
                    if (std.mem.endsWith(u8, set, ".tar.gz")) {
                        try spawnAndWait(&[_][]const u8{
                            "tar",
                            "-xzf",
                            file_path,
                            "-C",
                            dataset_dir_str,
                        }, allocator);
                    } else if (std.mem.endsWith(u8, set, ".gz")) {
                        try spawnAndWait(&[_][]const u8{
                            "gzip",
                            "-d",
                            file_path,
                        }, allocator);
                    } else {
                        std.log.warn("Unknown file type for {s}, skipping extraction", .{set});
                        continue;
                    }
                }
            },
        }

        std.log.info("Moving vector files to dataset root directory", .{});

        // Move every .fvecs, .ivecs, or .bvecs file potentially nested in dataset dir
        // to the dataset dir root
        var walker = try dataset_dir.walk(allocator);
        defer walker.deinit();

        out: while (walker.next() catch |e| {
            try stderr.print("Error walking dataset directory, you can try moving the vector files yourself: {}\n", .{e});
            try stderr.flush();
            std.process.exit(0);
        }) |entry| {
            if (entry.kind == .file) {
                std.log.debug("Found file: {s}", .{entry.path});
                const ext = std.fs.path.extension(entry.path);
                for (valid_extensions) |valid_ext| {
                    if (std.mem.eql(u8, ext, valid_ext)) {
                        const file_name = std.fs.path.basename(entry.path);
                        std.log.info("Moving {s} to {s}", .{ entry.path, file_name });
                        dataset_dir.rename(entry.path, file_name) catch |e| {
                            try stderr.print("Error moving file, trying to move other files: {}\n", .{e});
                            try stderr.flush();
                            continue :out;
                        };
                        break;
                    }
                }
            }
        }

        std.log.info("Finished installing the dataset", .{});
    }
};

fn usage(exe_name: []const u8) std.io.Writer.Error!void {
    try stdout.print("Usage: {s} [dataset_name]\n", .{exe_name});
    try stdout.print("Available datasets:\n", .{});
    inline for (std.meta.fieldNames(VectorSet)) |name| {
        try stdout.print("  {s}\n", .{name});
    }
    try stdout.flush();
    std.process.exit(0);
}

fn checkHelp(arg: [:0]const u8, exe_name: []const u8) std.io.Writer.Error!void {
    const help_args = [_][]const u8{ "help", "--help", "-h" };
    for (help_args) |help_arg| {
        if (std.mem.eql(u8, arg, help_arg)) {
            try usage(exe_name);
            return;
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = std.process.args();
    const exe_path = args.next() orelse @src().file;
    const exe_name = std.fs.path.basename(exe_path);

    if (args.next()) |dataset_name| {
        try checkHelp(dataset_name, exe_name);

        const vector_set = std.meta.stringToEnum(VectorSet, dataset_name) orelse {
            try stderr.print("Error: Unknown dataset '{s}'.\n", .{dataset_name});
            try stderr.flush();
            try usage(exe_name);
            std.process.exit(1);
        };

        try vector_set.install(allocator);
        return;
    }

    try stdout.print("Welcome to TEXMEXSTEAL!\n", .{});
    try stdout.print("Install the vector set of your choosing:\n", .{});
    inline for (std.meta.fieldNames(VectorSet), 1..) |name, i| {
        try stdout.print("  {d}: {s}\n", .{ i, name });
    }
    try stdout.flush();

    const vector_set: VectorSet = while (true) {
        try stdout.print("Your choice: ", .{});
        try stdout.flush();
        const input = stdin.takeDelimiter('\n') catch |e| switch (e) {
            error.StreamTooLong => {
                try stderr.print("Invalid input (too long). Try again.\n", .{});
                continue;
            },
            error.ReadFailed => {
                try stderr.print("Error reading input. Exiting.\n", .{});
                return;
            },
        } orelse {
            try stderr.print("\nEnd-of-input detected. Exiting.\n", .{});
            return;
        };

        const trimmed_input = std.mem.trim(u8, input, " \t\r\n");
        if (trimmed_input.len == 0) {
            try stderr.print("No input detected. Please enter a number.\n", .{});
            continue;
        }
        const choice = std.fmt.parseInt(u8, trimmed_input, 10) catch |e| {
            switch (e) {
                error.Overflow => {
                    try stderr.print("Input number is too large. Try again.\n", .{});
                },
                error.InvalidCharacter => {
                    try stderr.print("Invalid input. Please enter a number.\n", .{});
                },
            }
            continue;
        };

        switch (choice) {
            1 => break .ANN_SIFT10K,
            2 => break .ANN_SIFT1M,
            3 => break .ANN_GIST1M,
            4 => break .ANN_SIFT1B,
            else => {
                try stderr.print("Invalid choice. Please enter a number between 1 and 4.\n", .{});
                continue;
            },
        }
    };

    try vector_set.install(allocator);
}
