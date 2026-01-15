const std = @import("std");
const zagra = @import("zagra");

const Vec = zagra.Vector(f32, 512);

pub fn main() !void {
    std.debug.print("ZAGRA executable\n", .{});
}
