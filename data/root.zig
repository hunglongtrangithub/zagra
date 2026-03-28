const ftp = @import("ftp.zig");
const vecs_to_npy = @import("vecs_to_npy.zig");
const vector_set = @import("vector_set.zig");

pub const VectorSet = vector_set.VectorSet;

test {
    _ = ftp;
    _ = vecs_to_npy;
    _ = vector_set;
}
