const std = @import("std");
const zagra = @import("zagra");

const Vec = zagra.Vector(f32, 512);

pub fn main() !void {
    std.debug.print("ZAGRA executable\n", .{});

    const v1 = Vec{ .data = undefined };
    const v2 = Vec{ .data = [_]f32{1} ** 512 };

    const sqdist = Vec.sqdist(&v1, &v2);

    std.debug.print("Squared distance between v1 and v2: {d}\n", .{sqdist});
}
