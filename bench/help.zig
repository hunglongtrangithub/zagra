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
pub fn checkHelp(arg: ?[:0]const u8, exe_path: [:0]const u8) void {
    if (arg) |a| {
        if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
            const exe_name = std.fs.path.basename(exe_path[0..]);
            std.debug.print(HELP_TEMPLATE, .{ exe_name, exe_name });
            std.process.exit(0);
        }
    }
}
