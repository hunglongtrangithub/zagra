const std = @import("std");

pub fn main() !void {
    std.debug.print("This is Zagra!\n", .{});
    std.debug.print("{}\n", .{@sizeOf(?usize)});
    std.debug.print("{}\n", .{@sizeOf(usize)});
}
