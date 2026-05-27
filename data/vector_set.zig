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

    fn checkExecutable(name: []const u8, io: std.Io, allocator: std.mem.Allocator) void {
        const run_result = std.process.run(allocator, io, .{
            .argv = &[_][]const u8{ "which", name },
        }) catch |e| {
            std.debug.print("Error starting 'which {s}': {}\n", .{ name, e });
            std.process.exit(1);
        };
        allocator.free(run_result.stdout);
        allocator.free(run_result.stderr);
        if (run_result.term.exited != 0) {
            std.debug.print("Error: '{s}' is not installed or not in PATH.\n", .{name});
            std.process.exit(1);
        }
    }

    fn spawn(io: std.Io, args: []const []const u8) std.process.Child {
        return std.process.spawn(io, .{
            .argv = args,
        }) catch |e| {
            std.debug.print("Error starting process: ", .{});
            for (args) |arg| std.debug.print("{s} ", .{arg});
            std.debug.print("\n{}\n", .{e});
            std.process.exit(1);
        };
    }

    pub fn install(
        self: VectorSet,
        allocator: std.mem.Allocator,
        io: std.Io,
        data_dir: []const u8,
    ) std.mem.Allocator.Error!void {
        const executables = [_][]const u8{ "tar", "gzip" };
        for (executables) |exe| {
            checkExecutable(exe, io, allocator);
        }

        const cwd = std.Io.Dir.cwd();

        cwd.createDir(io, data_dir, .default_dir) catch |e| switch (e) {
            error.PathAlreadyExists => log.info("Data directory already exists: {s}", .{data_dir}),
            else => {
                std.debug.print("Error creating data directory: {}\n", .{e});
                std.process.exit(1);
            },
        };

        const dataset_dir_str = std.fs.path.join(allocator, &[_][]const u8{ data_dir, @tagName(self) }) catch
            return std.mem.Allocator.Error.OutOfMemory;
        defer allocator.free(dataset_dir_str);
        cwd.createDir(io, dataset_dir_str, .default_dir) catch |e| switch (e) {
            error.PathAlreadyExists => log.warn("Vector set directory {s} already exists", .{@tagName(self)}),
            else => {
                std.debug.print("Error creating dataset directory: {}\n", .{e});
                std.process.exit(1);
            },
        };

        switch (self) {
            .ANN_SIFT10K, .ANN_SIFT1M, .ANN_GIST1M => {
                const file_name = switch (self) {
                    .ANN_SIFT10K => "siftsmall.tar.gz",
                    .ANN_SIFT1M => "sift.tar.gz",
                    .ANN_GIST1M => "gist.tar.gz",
                    else => unreachable,
                };

                const tar_file_path = std.fs.path.join(allocator, &[_][]const u8{ dataset_dir_str, file_name }) catch
                    return std.mem.Allocator.Error.OutOfMemory;
                defer allocator.free(tar_file_path);

                const tar_file_url = try std.fmt.allocPrint(allocator, "{s}{s}", .{ URL_PREFIX, file_name });
                defer allocator.free(tar_file_url);

                log.info("Downloading {s}...", .{file_name});
                const results = ftp.downloadFiles(
                    io,
                    allocator,
                    &[_]ftp.DownloadItem{.{
                        .url = tar_file_url,
                        .output_path = tar_file_path,
                    }},
                ) catch |e| {
                    switch (e) {
                        error.OutOfMemory => std.debug.print("Out of memory.", .{}),
                        error.TooManyFiles => std.debug.print("Too many files to download (max 255).", .{}),
                    }
                    std.debug.print(" Exiting.\n", .{});
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
                const extract_cmd = &[_][]const u8{
                    "tar",
                    "-xzf",
                    tar_file_path,
                    "-C",
                    dataset_dir_str,
                };
                var child_process = spawn(io, extract_cmd);
                const wait_result = child_process.wait(io) catch |e| {
                    std.debug.print("Error waiting for process: ", .{});
                    for (extract_cmd) |arg| std.debug.print("{s} ", .{arg});
                    std.debug.print("\n{}\n", .{e});
                    std.process.exit(1);
                };
                if (wait_result.exited != 0) {
                    std.debug.print("Process exited with code {}: ", .{wait_result.exited});
                    for (extract_cmd) |arg| std.debug.print("{s} ", .{arg});
                    std.debug.print("\n", .{});
                    std.process.exit(1);
                }
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
                    file_paths[i] = std.fs.path.join(allocator, &[_][]const u8{ dataset_dir_str, name }) catch
                        return std.mem.Allocator.Error.OutOfMemory;
                    download_items[i] = .{
                        .url = URL_PREFIX ++ name,
                        .output_path = file_paths[i],
                    };
                }

                const results = ftp.downloadFiles(
                    io,
                    allocator,
                    &download_items,
                ) catch |e| {
                    switch (e) {
                        error.OutOfMemory => std.debug.print("Out of memory.", .{}),
                        error.TooManyFiles => std.debug.print("Too many files to download (max 255).", .{}),
                    }
                    std.debug.print(" Exiting.\n", .{});
                    std.process.exit(1);
                };
                defer allocator.free(results);

                for (results) |result| {
                    _ = result catch |err| {
                        std.debug.print("Some downloads failed: {}. Exiting.\n", .{err});
                        std.process.exit(1);
                    };
                }

                var child_processes: [4]std.process.Child = undefined;
                for (file_names, file_paths, &child_processes) |name, file_path, *child| {
                    log.info("Extracting {s}...", .{name});
                    const cmd = if (std.mem.endsWith(u8, name, ".tar.gz"))
                        &[_][]const u8{
                            "tar",
                            "-xzf",
                            file_path,
                            "-C",
                            dataset_dir_str,
                        }
                    else if (std.mem.endsWith(u8, name, ".gz"))
                        &[_][]const u8{
                            "gzip",
                            "-df",
                            file_path,
                        }
                    else {
                        log.warn("Unknown file type for {s}, skipping extraction", .{name});
                        continue;
                    };
                    child.* = spawn(io, cmd);
                }

                var all_success = true;
                for (&child_processes, 0..) |*child, i| {
                    const result = child.wait(io);
                    if (result) |wait_result| {
                        if (wait_result.exited == 0) {
                            std.debug.print("Extract success: {s}\n", .{file_names[i]});
                        } else {
                            std.debug.print("Non-zero return code for {s}: {d}\n", .{ file_names[i], wait_result.exited });
                            all_success = false;
                            break;
                        }
                    } else |err| {
                        std.debug.print("Error waiting for extraction on {s}: {}\n", .{ file_names[i], err });
                        all_success = false;
                        break;
                    }
                }

                if (!all_success) {
                    std.debug.print("Some files failed to extract. Stopping process.\n", .{});
                    std.process.exit(1);
                } else {
                    std.debug.print("All files extracted successfully.\n", .{});
                }
            },
        }

        log.info("Moving vector files to dataset root directory", .{});

        var dataset_dir = cwd.openDir(
            io,
            dataset_dir_str,
            .{ .iterate = true },
        ) catch |e| {
            std.debug.print("Error opening dataset directory: {}\n", .{e});
            std.process.exit(1);
        };
        defer dataset_dir.close(io);
        var walker = try dataset_dir.walk(allocator);
        defer walker.deinit();

        while (walker.next(io) catch |e| {
            std.debug.print("Error walking dataset directory, you can try moving the vector files yourself: {}\n", .{e});
            std.process.exit(0);
        }) |entry| {
            if (entry.kind != .file) continue;
            const ext = std.fs.path.extension(entry.path);
            const vecs_type = vecs_to_npy.VecsType.fromExtension(ext) orelse continue;
            log.debug("Found valid file with vecs type {}: {s}", .{ vecs_type, entry.path });

            const file_name = std.fs.path.basename(entry.path);
            log.info("Moving {s} to {s}", .{ entry.path, file_name });
            dataset_dir.rename(
                entry.path,
                dataset_dir,
                file_name,
                io,
            ) catch |e| {
                std.debug.print("Error moving file, trying to move other files: {}\n", .{e});
                continue;
            };
        }

        log.info("Finished installing the dataset", .{});
    }
};
