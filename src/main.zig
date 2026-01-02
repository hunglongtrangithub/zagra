const std = @import("std");
const zagra = @import("zagra");

const Vec3f = zagra.Vector(.f32, .D3);

pub fn main() !void {
    std.debug.print("ZAGRA executable\n", .{});

    const v1 = Vec3f.init([3]f32{ 1, 2, 3 });
    const v2 = Vec3f.init([3]f32{ 6, 22, 43 });
    const sqdist = v1.sqdist(&v2);

    std.debug.print("Squared Distance: {d}\n", .{sqdist});
}
