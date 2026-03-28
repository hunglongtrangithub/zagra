const std = @import("std");
const ftp = @import("ftp.zig");
const vecs_to_npy = @import("vecs_to_npy.zig");

const log = std.log.scoped(.vector_set);

pub const VectorSet = enum {
    ANN_SIFT10K,
    ANN_SIFT1M,
    ANN_GIST1M,
    ANN_SIFT1B,

    const URL_PREFIX = "ftp://ftp.irisa.fr/local/texmex/corpus/";

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

    fn makeDatasetDir(dataset_dir_str: []const u8) std.fs.Dir {
        if (std.fs.path.isAbsolute(dataset_dir_str)) {
            std.fs.makeDirAbsolute(dataset_dir_str) catch |e| switch (e) {
                error.PathAlreadyExists => {},
                else => {
                    std.debug.print("Error creating dataset directory: {}\n", .{e});
                    std.process.exit(1);
                },
            };
        } else {
            std.fs.cwd().makeDir(dataset_dir_str) catch |e| switch (e) {
                error.PathAlreadyExists => {},
                else => {
                    std.debug.print("Error creating dataset directory: {}\n", .{e});
                    std.process.exit(1);
                },
            };
        }

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
        const executables = [_][]const u8{ "tar", "gzip" };
        for (executables) |exe| {
            checkExecutable(exe, allocator);
        }

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
                            "-df",
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
