const std = @import("std");
const zagra = @import("zagra");

pub fn main() !void {
    std.debug.print("ZAGRA executable\n", .{});
    const Vec3f = zagra.Vector(i32, 3);

    const v1 = Vec3f.init([3]i32{ 1, 2, 3 });
    const v2 = Vec3f.init([3]i32{ 6, 22, 43 });

    const sqdist = Vec3f.sqdist(&v1, &v2);

    std.debug.print("Squared Distance: {d}\n", .{sqdist});
}
