const std = @import("std");
const config = @import("config");
const vector_set_mod = @import("vector_set.zig");
const vecs_to_npy_mod = @import("vecs_to_npy.zig");

const log = std.log.scoped(.main);

const VectorSet = vector_set_mod.VectorSet;

fn usage(stdout: *std.Io.Writer, exe_name: []const u8) std.Io.Writer.Error!void {
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

fn checkHelp(stdout: *std.Io.Writer, arg: [:0]const u8, exe_name: []const u8) std.Io.Writer.Error!void {
    const help_args = [_][]const u8{ "--help", "-h" };
    for (help_args) |help_arg| {
        if (std.mem.eql(u8, arg, help_arg)) {
            try usage(stdout, exe_name);
            return;
        }
    }
}

pub fn main() (std.mem.Allocator.Error || std.Io.Writer.Error)!void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    const exe_path = args.next() orelse @src().file;
    const exe_name = std.fs.path.basename(exe_path);

    var stdin_buffer: [1024]u8 = undefined;
    var stdin_reader = std.fs.File.stdin().reader(&stdin_buffer);
    const stdin = &stdin_reader.interface;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    if (args.next()) |dataset_name| {
        try checkHelp(stdout, dataset_name, exe_name);

        const vector_set = std.meta.stringToEnum(VectorSet, dataset_name) orelse {
            std.debug.print("Error: Unknown dataset '{s}'.\n", .{dataset_name});
            try usage(stdout, exe_name);
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
                try usage(stdout, exe_name);
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
            vecs_to_npy_mod.convertVecsToNpy(dataset_dir);
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
    vecs_to_npy_mod.convertVecsToNpy(dataset_dir);
}
