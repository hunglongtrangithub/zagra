const std = @import("std");
const builtin = @import("builtin");

const znpy = @import("znpy");
const FromFileReaderError = znpy.array.static.FromFileReaderError;

const mod_vector = @import("vector.zig");
const Vector = mod_vector.Vector;

pub fn Dataset(comptime T: type, comptime N: usize) type {
    const Vec = Vector(T, N);
    const Array = znpy.array.static.StaticArray(T, 2);

    return struct {
        array: Array,
        len: usize,

        const Self = @This();

        /// Load a dataset of fixed-size vectors from a .npy file reader.
        /// The .npy file must contain a 2D array where one dimension is of size N.
        /// Return an error if the shape is invalid. A shape is valid if:
        /// - The array is 2-dimensional.
        /// - For C-order arrays, the second dimension is N.
        /// - For F-order arrays, the first dimension is N.
        pub fn fromNpyFileReader(reader: *std.io.Reader, allocator: std.mem.Allocator) (FromFileReaderError || error.InvalidShape)!Self {
            const array = try Array.fromFileAlloc(reader, allocator);

            switch (array.shape.order) {
                .C => {
                    if (array.shape.dims[1] != N) {
                        return error.InvalidShape;
                    }

                    return Self{
                        .array = array,
                        .len = array.shape.dims[0],
                    };
                },
                .F => {
                    if (array.shape.dims[0] != N) {
                        return error.InvalidShape;
                    }

                    return Self{
                        .array = array,
                        .len = array.shape.dims[1],
                    };
                },
            }

            return Self{
                .array = array,
            };
        }

        /// Get the vector at the specified index.
        pub fn get(self: *Self, index: usize) ?Vec {
            if (index >= self.len) {
                return null;
            }

            // SAFETY: The vector data is contiguous in memory and has length N,
            // as enforced by fromNpyFileReader. The bounnds are already checked,
            // which means this slice is within the data buffer.
            const data = self.array.data_buffer[index * N .. index * N + N];
            return Vec{ .data = data };
        }
    };
}
