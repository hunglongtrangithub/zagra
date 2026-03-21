const std = @import("std");

const HELP_TEMPLATE =
    \\Usage: {s} [prefix]
    \\
    \\Optional argument 'prefix' will be used to prefix CSV result files.
    \\Example: {s} myrun -> myrun_<name>.csv
    \\
    \\Available flags:
    \\  --help, -h     Show this help message
    \\
;

/// Checks if the provided argument is a help flag and prints usage information if it is.
pub fn checkHelp(
    stdout: *std.Io.Writer,
    arg: ?[:0]const u8,
    exe_path: [:0]const u8,
) std.Io.Writer.Error!void {
    if (arg) |a| {
        const help_args = [_][]const u8{ "--help", "-h" };
        for (help_args) |help_arg| {
            if (std.mem.eql(u8, a, help_arg)) {
                const exe_name = std.fs.path.basename(exe_path[0..]);
                try stdout.print(HELP_TEMPLATE, .{ exe_name, exe_name });
                try stdout.flush();
                std.process.exit(0);
            }
        }
    }
}
