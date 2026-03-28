const std = @import("std");

const DEFAULT_USAGE =
    \\Usage: {s} [prefix]
    \\
    \\Optional argument 'prefix' will be used to prefix CSV result files.
    \\Example: {s} myrun -> myrun_<name>.csv
    \\
    \\Available flags:
    \\  --help, -h     Show this help message
    \\
;

/// Checks if the provided argument is a help flag and prints default usage information if it is.
pub fn checkHelp(
    stdout: *std.Io.Writer,
    arg: ?[:0]const u8,
    exe_path: [:0]const u8,
) std.Io.Writer.Error!void {
    return checkHelpWithUsage(stdout, arg, exe_path, null);
}

/// Variant that accepts an optional custom usage string.
/// If `custom_usage` is provided, it will be printed instead of the default HELP_TEMPLATE.
/// `custom_usage` may contain a single `{s}` place-holder for the exe name.
pub fn checkHelpWithUsage(
    stdout: *std.Io.Writer,
    arg: ?[:0]const u8,
    exe_path: [:0]const u8,
    comptime custom_usage: ?[:0]const u8,
) std.Io.Writer.Error!void {
    const a = arg orelse return;
    const help_args = [_][]const u8{ "--help", "-h" };
    for (help_args) |help_arg| {
        if (std.mem.eql(u8, a, help_arg)) {
            const exe_name = std.fs.path.basename(exe_path[0..]);
            if (custom_usage) |u| {
                try stdout.print(u, .{exe_name});
            } else {
                try stdout.print(DEFAULT_USAGE, .{ exe_name, exe_name });
            }
            try stdout.flush();
            std.process.exit(0);
        }
    }
}
