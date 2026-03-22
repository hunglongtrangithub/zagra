const std = @import("std");
const config = @import("config");
const ftp = @import("ftp.zig");
const vecs_to_npy = @import("vecs_to_npy.zig");

const log = std.log.scoped(.main);

var stdin_buffer: [1024]u8 = undefined;
var stdin_reader = std.fs.File.stdin().reader(&stdin_buffer);
const stdin = &stdin_reader.interface;

var stdout_buffer: [1024]u8 = undefined;
var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
const stdout = &stdout_writer.interface;

const URL_PREFIX = "ftp://ftp.irisa.fr/local/texmex/corpus/";

const VectorSet = enum {
    ANN_SIFT10K,
    ANN_SIFT1M,
    ANN_GIST1M,
    ANN_SIFT1B,

    fn checkExecutable(name: []const u8, allocator: std.mem.Allocator) void {
        const run_result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{ "which", name },
        }) catch |e| {
            std.debug.print("Error starting 'which {s}': {}\n", .{ name, e });
            std.process.exit(1);
        };
        allocator.free(run_result.stdout);
        allocator.free(run_result.stderr);
        if (run_result.term.Exited != 0) {
            std.debug.print("Error: '{s}' is not installed or not in PATH.\n", .{name});
            std.process.exit(1);
        }
    }

    /// Spawns a child process with the given arguments.
    /// Exits on spawn failure. Return the spawned child process on success.
    fn spawn(
        argv: []const []const u8,
        allocator: std.mem.Allocator,
    ) std.process.Child {
        var cmd = std.process.Child.init(argv, allocator);
        cmd.spawn() catch |e| {
            std.debug.print("Error starting process: {}\n", .{e});
            std.debug.print("Argv: ", .{});
            for (argv) |arg| std.debug.print("{s} ", .{arg});
            std.debug.print("\n", .{});
            std.process.exit(1);
        };
        return cmd;
    }

    /// Waits for the given child process to finish.
    /// Exits if the process fails or exits with a non-zero code.
    fn wait(cmd: *std.process.Child) void {
        const term = cmd.wait() catch |e| {
            std.debug.print("Error waiting for process: {}\n", .{e});
            std.debug.print("Argv: ", .{});
            for (cmd.argv) |arg| std.debug.print("{s} ", .{arg});
            std.debug.print("\n", .{});
            std.process.exit(1);
        };
        if (term.Exited != 0) {
            std.debug.print("Error: Process exited with code {d}.\n", .{term.Exited});
            std.debug.print("Argv: ", .{});
            for (cmd.argv) |arg| std.debug.print("{s} ", .{arg});
            std.debug.print("\n", .{});
            std.process.exit(1);
        }
    }

    /// Make dataset directory if it doesn't exist, or verify access if it does. Exits on failure.
    /// Returns the opened directory on success.
    fn makeDatasetDir(dataset_dir_str: []const u8) std.fs.Dir {
        if (std.fs.path.isAbsolute(dataset_dir_str)) {
            std.fs.makeDirAbsolute(dataset_dir_str) catch |e| switch (e) {
                error.PathAlreadyExists => log.info("Dataset directory already exists: {s}", .{dataset_dir_str}),
                else => {
                    std.debug.print("Error creating dataset directory: {}\n", .{e});
                    std.process.exit(1);
                },
            };
        } else {
            std.fs.cwd().makeDir(dataset_dir_str) catch |e| switch (e) {
                error.PathAlreadyExists => log.info("Dataset directory already exists: {s}", .{dataset_dir_str}),
                else => {
                    std.debug.print("Error creating dataset directory: {}\n", .{e});
                    std.process.exit(1);
                },
            };
        }

        // Try opening the directory
        return std.fs.cwd().openDir(dataset_dir_str, .{ .iterate = true }) catch |e| {
            std.debug.print("Error opening dataset directory: {}\n", .{e});
            std.process.exit(1);
        };
    }

    pub fn install(
        self: VectorSet,
        allocator: std.mem.Allocator,
        data_dir: []const u8,
    ) std.mem.Allocator.Error!void {
        // Check if executables are available
        const executables = [_][]const u8{ "tar", "gzip" };
        for (executables) |exe| {
            checkExecutable(exe, allocator);
        }

        // Ensure parent data directory exists (e.g., "data/")
        if (std.fs.path.isAbsolute(data_dir)) {
            std.fs.makeDirAbsolute(data_dir) catch |e| switch (e) {
                error.PathAlreadyExists => log.info("Data directory already exists: {s}", .{data_dir}),
                else => {
                    std.debug.print("Error creating data directory: {}\n", .{e});
                    std.process.exit(1);
                },
            };
        } else {
            std.fs.cwd().makePath(data_dir) catch |e| switch (e) {
                error.PathAlreadyExists => log.info("Data directory already exists: {s}", .{data_dir}),
                else => {
                    std.debug.print("Error creating data directory: {}\n", .{e});
                    std.process.exit(1);
                },
            };
        }
        const dataset_dir_str = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ data_dir, @tagName(self) });
        defer allocator.free(dataset_dir_str);
        var dataset_dir = makeDatasetDir(dataset_dir_str);
        defer dataset_dir.close();

        switch (self) {
            .ANN_SIFT10K, .ANN_SIFT1M, .ANN_GIST1M => {
                const file_name = switch (self) {
                    .ANN_SIFT10K => "siftsmall.tar.gz",
                    .ANN_SIFT1M => "sift.tar.gz",
                    .ANN_GIST1M => "gist.tar.gz",
                    else => unreachable,
                };

                const tar_file_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dataset_dir_str, file_name });
                defer allocator.free(tar_file_path);

                const tar_file_url = try std.fmt.allocPrint(allocator, "{s}{s}", .{ URL_PREFIX, file_name });
                defer allocator.free(tar_file_url);

                log.info("Downloading {s}...", .{file_name});
                const results = ftp.downloadFiles(
                    allocator,
                    &[_]ftp.DownloadItem{.{
                        .url = tar_file_url,
                        .output_path = tar_file_path,
                    }},
                ) orelse {
                    std.debug.print("Download failed (out of memory). Exiting.\n", .{});
                    std.process.exit(1);
                };
                defer allocator.free(results);

                for (results) |result| {
                    _ = result catch |err| {
                        std.debug.print("Download failed: {}. Exiting.\n", .{err});
                        std.process.exit(1);
                    };
                }

                log.info("Extracting...", .{});
                var cmd = spawn(&[_][]const u8{
                    "tar",
                    "-xzf",
                    tar_file_path,
                    "-C",
                    dataset_dir_str,
                }, allocator);
                wait(&cmd);
            },
            .ANN_SIFT1B => {
                const file_names = [_][]const u8{
                    "bigann_base.bvecs.gz",
                    "bigann_learn.bvecs.gz",
                    "bigann_query.bvecs.gz",
                    "bigann_gnd.tar.gz",
                };

                const file_paths = try allocator.alloc([]const u8, file_names.len);
                defer {
                    for (file_paths) |path| allocator.free(path);
                    allocator.free(file_paths);
                }

                var download_items: [file_names.len]ftp.DownloadItem = undefined;
                inline for (file_names, 0..) |name, i| {
                    file_paths[i] = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dataset_dir_str, name });
                    download_items[i] = .{
                        .url = URL_PREFIX ++ name,
                        .output_path = file_paths[i],
                    };
                }

                // Download all files in parallel
                const results = ftp.downloadFiles(allocator, &download_items) orelse {
                    std.debug.print("Download failed (out of memory). Exiting.\n", .{});
                    std.process.exit(1);
                };
                defer allocator.free(results);

                for (results) |result| {
                    _ = result catch |err| {
                        std.debug.print("Some downloads failed: {}. Exiting.\n", .{err});
                        std.process.exit(1);
                    };
                }

                var cmds: [4]std.process.Child = undefined;
                // Extract the ground truth set with tar, and all other sets with gzip, in parallel
                for (file_names, file_paths, &cmds) |name, file_path, *cmd| {
                    log.info("Extracting {s}...", .{name});
                    cmd.* = if (std.mem.endsWith(u8, name, ".tar.gz"))
                        spawn(&[_][]const u8{
                            "tar",
                            "-xzf",
                            file_path,
                            "-C",
                            dataset_dir_str,
                        }, allocator)
                    else if (std.mem.endsWith(u8, name, ".gz"))
                        spawn(&[_][]const u8{
                            "gzip",
                            "-d",
                            file_path,
                        }, allocator)
                    else {
                        log.warn("Unknown file type for {s}, skipping extraction", .{name});
                        continue;
                    };
                }
                for (&cmds) |*cmd| wait(cmd);
            },
        }

        log.info("Moving vector files to dataset root directory", .{});

        // Move every .fvecs, .ivecs, or .bvecs file potentially nested in dataset dir
        // to the dataset dir root
        var walker = try dataset_dir.walk(allocator);
        defer walker.deinit();

        while (walker.next() catch |e| {
            std.debug.print("Error walking dataset directory, you can try moving the vector files yourself: {}\n", .{e});
            std.process.exit(0);
        }) |entry| {
            if (entry.kind != .file) continue;
            const ext = std.fs.path.extension(entry.path);
            const vecs_type = vecs_to_npy.VecsType.fromExtension(ext) orelse continue;
            log.debug("Found valid file with vecs type {}: {s}", .{ vecs_type, entry.path });

            const file_name = std.fs.path.basename(entry.path);
            log.info("Moving {s} to {s}", .{ entry.path, file_name });
            dataset_dir.rename(entry.path, file_name) catch |e| {
                std.debug.print("Error moving file, trying to move other files: {}\n", .{e});
                continue;
            };
        }

        log.info("Finished installing the dataset", .{});
    }
};

fn usage(exe_name: []const u8) std.Io.Writer.Error!void {
    try stdout.print("Usage: {s} [dataset_name] [custom_data_dir] [--no-convert] [--help|-h]\n", .{exe_name});
    try stdout.print("If dataset_name is not provided, enter interactive application.\n", .{});
    try stdout.print("If custom_data_dir is not provided, default to ./{s}\n", .{config.DATA_DIR});
    try stdout.print("If --no-convert is provided, do not convert fvecs/bvecs/ivecs files to npy files.\n", .{});
    try stdout.print("Available datasets:\n", .{});
    inline for (std.meta.fieldNames(VectorSet)) |name| {
        try stdout.print("  {s}\n", .{name});
    }
    try stdout.flush();
    std.process.exit(0);
}

fn checkHelp(arg: [:0]const u8, exe_name: []const u8) std.Io.Writer.Error!void {
    const help_args = [_][]const u8{ "--help", "-h" };
    for (help_args) |help_arg| {
        if (std.mem.eql(u8, arg, help_arg)) {
            try usage(exe_name);
            return;
        }
    }
}

pub fn main() (std.mem.Allocator.Error || std.Io.Writer.Error)!void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = std.process.args();
    const exe_path = args.next() orelse @src().file;
    const exe_name = std.fs.path.basename(exe_path);

    if (args.next()) |dataset_name| {
        try checkHelp(dataset_name, exe_name);

        const vector_set = std.meta.stringToEnum(VectorSet, dataset_name) orelse {
            std.debug.print("Error: Unknown dataset '{s}'.\n", .{dataset_name});
            try usage(exe_name);
            std.process.exit(1);
        };

        const data_dir_input = args.next() orelse config.DATA_DIR;
        const trimmed_data_dir = std.mem.trim(u8, data_dir_input, " \t\r\n");
        const final_data_dir = if (trimmed_data_dir.len == 0) config.DATA_DIR else trimmed_data_dir;

        // Check conversion flag
        const do_convert = if (args.next()) |flag| blk: {
            if (std.mem.eql(u8, flag, "--no-convert")) {
                break :blk false;
            } else {
                std.debug.print("Unknown flag '{s}'.\n", .{flag});
                try usage(exe_name);
                std.process.exit(1);
            }
        } else true;

        try vector_set.install(allocator, final_data_dir);

        if (do_convert) {
            const dataset_dir_str = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ final_data_dir, @tagName(vector_set) });
            defer allocator.free(dataset_dir_str);
            var dataset_dir = std.fs.cwd().openDir(dataset_dir_str, .{ .iterate = true }) catch |e| {
                std.debug.print("Error opening dataset directory {s}: {}\n", .{ dataset_dir_str, e });
                std.process.exit(1);
            };
            defer dataset_dir.close();

            log.info("Converting vector files to .npy format...", .{});
            vecs_to_npy.convertVecsToNpy(dataset_dir);
        } else {
            log.info("Skipping conversion to .npy format as per --no-convert flag.", .{});
        }
        return;
    }

    // Ask for vector set
    try stdout.print("Welcome to TEXMEXSTEAL!\n", .{});
    try stdout.print("Install the vector set of your choosing (send end-of-file signal to exit):\n", .{});
    inline for (std.meta.fieldNames(VectorSet), 1..) |name, i| {
        try stdout.print("  {d}: {s}\n", .{ i, name });
    }
    try stdout.flush();
    const vector_set: VectorSet = while (true) {
        try stdout.print("Your choice: ", .{});
        try stdout.flush();
        const input = stdin.takeDelimiter('\n') catch |e| switch (e) {
            error.StreamTooLong => {
                std.debug.print("Invalid input (too long). Try again.\n", .{});
                continue;
            },
            error.ReadFailed => {
                std.debug.print("Error reading input. Exiting.\n", .{});
                return;
            },
        } orelse {
            std.debug.print("\nEnd-of-input detected. Exiting.\n", .{});
            return;
        };

        const trimmed_input = std.mem.trim(u8, input, " \t\r\n");
        if (trimmed_input.len == 0) {
            std.debug.print("No input detected. Please enter a number.\n", .{});
            continue;
        }
        const choice = std.fmt.parseInt(u8, trimmed_input, 10) catch |e| {
            switch (e) {
                error.Overflow => {
                    std.debug.print("Invalid number. Try again.\n", .{});
                },
                error.InvalidCharacter => {
                    std.debug.print("Invalid input. Please enter a number.\n", .{});
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
                std.debug.print("Invalid choice. Please enter a number between 1 and 4.\n", .{});
                continue;
            },
        }
    };

    // Ask for data directory
    try stdout.print("Enter data directory (either relative to CWD or absolute) (press Enter to use default): ", .{});
    try stdout.flush();
    const data_dir_input = stdin.takeDelimiter('\n') catch |e| blk: {
        switch (e) {
            error.StreamTooLong => std.debug.print("Invalid input (too long). Using default directory.\n", .{}),
            error.ReadFailed => std.debug.print("Error reading input. Using default directory.\n", .{}),
        }
        break :blk config.DATA_DIR;
    } orelse {
        std.debug.print("\nEnd-of-input detected. Exiting.\n", .{});
        return;
    };

    // Ask about conversion to .npy
    try stdout.print("Convert .fvecs/.ivecs/.bvecs files to .npy format? (press Enter to convert, anything else to skip): ", .{});
    try stdout.flush();
    const response = stdin.takeDelimiter('\n') catch |e| {
        std.debug.print("Error reading input: {}. Exiting.\n", .{e});
        return;
    } orelse {
        std.debug.print("\nEnd-of-input detected. Exiting.\n", .{});
        return;
    };

    // Install the dataset
    const trimmed_data_dir = std.mem.trim(u8, data_dir_input, " \t\r\n");
    const final_data_dir = if (trimmed_data_dir.len == 0) config.DATA_DIR else trimmed_data_dir;
    try vector_set.install(allocator, final_data_dir);

    // Convert to .npy format if user agreed
    const do_convert = response.len == 0;
    if (!do_convert) {
        log.info("Skipping conversion to .npy format as per user choice.", .{});
        return;
    }
    const dataset_dir_str = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ final_data_dir, @tagName(vector_set) });
    defer allocator.free(dataset_dir_str);
    var dataset_dir = std.fs.cwd().openDir(dataset_dir_str, .{ .iterate = true }) catch |e| {
        std.debug.print("Error opening dataset directory {s}: {}\n", .{ dataset_dir_str, e });
        std.process.exit(1);
    };
    defer dataset_dir.close();

    log.info("Converting vector files to .npy format...", .{});
    vecs_to_npy.convertVecsToNpy(dataset_dir);
}

test {
    _ = ftp;
    _ = vecs_to_npy;
}
