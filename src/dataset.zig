const std = @import("std");
const builtin = @import("builtin");

const znpy = @import("znpy");
const FromFileReaderError = znpy.array.static.FromFileReaderError;

const mod_vector = @import("vector.zig");
const Vector = mod_vector.Vector;

/// A dataset of fixed-size vectors loaded from a .npy file.
pub fn Dataset(comptime T: type, comptime N: usize) type {
    const Vec = Vector(T, N);
    const Array = znpy.array.static.StaticArray(T, 2);

    return struct {
        /// Buffer containing the dataset's data.
        /// The data is stored as a flat array of length `len * N`.
        /// The vector at index `i` is stored in `data_buffer[i * N .. (i + 1) * N]`.
        /// The number of elements in the buffer is `len * N`, which is less than `std.math.maxInt(isize)`,
        /// as enforced by fromNpyFileReader.
        // The buffer is aligned to 64 bytes for SIMD performance.
        // With the current supported types and dimensions, this makes every
        // vector start at a 64-byte aligned address.
        data_buffer: []align(64) const T,
        /// Number of vectors in the dataset. Should be no more than `std.math.maxInt(isize) / N`.
        len: usize,

        const Self = @This();

        /// Load a dataset of fixed-size vectors from a .npy file reader.
        /// The .npy file must contain a 2D array where one dimension is of size N.
        /// Return an error if the shape is invalid. A shape is valid if:
        /// - The array is 2-dimensional.
        /// - For C-order arrays, the second dimension is N.
        /// - For F-order arrays, the first dimension is N.
        pub fn fromNpyFileReader(reader: *std.io.Reader, allocator: std.mem.Allocator) (FromFileReaderError || error.InvalidShape)!Self {
            const array = try Array.fromFileAllocAligned(
                reader,
                std.mem.Alignment.@"64",
                allocator,
            );

            const dataset_len = blk: switch (array.shape.order) {
                .C => {
                    if (array.shape.dims[1] != N) {
                        return error.InvalidShape;
                    }

                    break :blk array.shape.dims[0];
                },
                .F => {
                    if (array.shape.dims[0] != N) {
                        return error.InvalidShape;
                    }
                    break :blk array.shape.dims[1];
                },
            };

            return Self{
                .data_buffer = @as([]align(64) const T, @alignCast(array.data_buffer)),
                .len = dataset_len,
            };
        }

        /// Deinitialize the dataset and free its data buffer.
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data_buffer);
        }

        /// Get a const pointer to the vector at the specified index.
        pub fn get(self: *const Self, index: usize) ?*const Vec {
            if (index >= self.len) {
                return null;
            }

            // SAFETY: The vector data is contiguous in memory and has length N,
            // as enforced by fromNpyFileReader. The bounnds are already checked,
            // which means this slice is within the data buffer.
            return self.getUnchecked(index);
        }

        /// Get a const pointer to the vector at the specified index without bounds checking.
        /// SAFETY: Caller must ensure index is valid (i.e., less than `len`).
        pub fn getUnchecked(self: *const Self, index: usize) *const Vec {
            std.debug.assert(index < self.len);
            // We return the address of the data already living in the data buffer.
            // No new struct is created; no data is moved.
            return @ptrCast(&self.data_buffer[index * N]);
        }
    };
}
