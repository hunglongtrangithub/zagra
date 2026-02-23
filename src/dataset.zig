const std = @import("std");
const builtin = @import("builtin");

const znpy = @import("znpy");

const mod_vector = @import("vector.zig");

/// A dataset of fixed-size vectors loaded from a .npy file.
/// The dataset contains vectors of type `Vector(T, N)`, where `T` is the element type
/// and `N` is the dimension of each vector.
pub fn Dataset(comptime T: type, comptime N: usize) type {
    const Vec = mod_vector.Vector(T, N);

    return struct {
        /// Buffer containing the dataset's data.
        /// The data is stored as a flat array of length `len * N`.
        /// The vector at index `i` is stored in `data_buffer[i * N .. (i + 1) * N]`.
        /// The number of elements in the buffer is `len * N`, which is less than `std.math.maxInt(isize)`,
        /// as enforced by fromNpyFileReader.
        // The buffer is aligned to 64 bytes for SIMD performance.
        // With the current supported types and dimensions, this makes every
        // vector start at a 64-byte aligned address.
        // Alignment of 64 bytes satisfies all natural alignments of types we support.
        data_buffer: []align(64) const T,
        /// Number of vectors in the dataset.
        len: usize,

        const Self = @This();

        /// Save the dataset to a .npy file writer.
        /// Order is row-major (C order).
        pub fn toNpyFile(
            self: *const Self,
            writer: *std.io.Writer,
            allocator: std.mem.Allocator,
        ) !void {
            const header = znpy.header.Header{
                .shape = &[_]usize{ self.len, N },
                .descr = try znpy.ElementType.fromZigType(T),
                .order = znpy.Order.C,
            };
            try header.writeAll(writer, allocator);
            try writer.writeAll(std.mem.sliceAsBytes(self.data_buffer));
        }

        /// Load a dataset of fixed-size vectors from a .npy file reader.
        /// The .npy file must contain a 2D array where one dimension is of size N.
        pub fn fromNpyFileReader(
            reader: *std.io.Reader,
            allocator: std.mem.Allocator,
        ) (znpy.array.static.FromFileReaderError || error{InvalidShape})!Self {
            const array = try znpy.array.static.StaticArray(T, 2).fromFileAllocAligned(
                reader,
                std.mem.Alignment.@"64",
                allocator,
            );

            const dataset_len = try verifyShape(array.shape);

            return Self{
                .data_buffer = @as([]align(64) const T, @alignCast(array.data_buffer)),
                .len = dataset_len,
            };
        }

        /// Load a dataset of fixed-size vectors from a .npy file buffer in memory (using mmap or similar).
        /// The .npy file must contain a 2D array where one dimension is of size N.
        /// The dataset does not have to be deinitialized since the file buffer is managed by caller.
        pub fn fromNpyFileBuffer(
            file_buffer: []const u8,
            allocator: std.mem.Allocator,
        ) (znpy.array.static.FromFileBufferError || error{ InvalidShape, MisalignedData })!Self {
            const array = try znpy.array.static.ConstStaticArray(T, 2).fromFileBuffer(
                file_buffer,
                allocator,
            );

            const dataset_len = try verifyShape(array.shape);

            // Npy file specification already ensures data is aligned to 64 bytes, but we double-check here.
            if (!std.mem.isAligned(@intFromPtr(array.data_buffer.ptr), 64)) return error.MisalignedData;

            return Self{
                .data_buffer = @as([]align(64) const T, @alignCast(array.data_buffer)),
                .len = dataset_len,
            };
        }

        /// Verify that the given shape is compatible with vectors of dimension N.
        /// Return the number of vectors if valid, otherwise return error.InvalidShape.
        /// A shape is valid if:
        /// - The array is 2-dimensional.
        /// - For C-order arrays, the second dimension is N.
        /// - For F-order arrays, the first dimension is N.
        fn verifyShape(shape: znpy.shape.StaticShape(2)) error{InvalidShape}!usize {
            switch (shape.order) {
                .C => {
                    if (shape.dims[1] != N) {
                        return error.InvalidShape;
                    }
                    return shape.dims[0];
                },
                .F => {
                    if (shape.dims[0] != N) {
                        return error.InvalidShape;
                    }
                    return shape.dims[1];
                },
            }
        }

        /// Deinitialize the dataset and free its data buffer.
        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
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
            // SAFETY: This cast works because:
            // 1. for @alignCast: The base buffer is 64-byte aligned and each vector has a stride
            // of either 128, 256, or 512 (a multiple of 64). Thus the address at self.data_buffer[index * N]
            // has alignment of 64.
            // 2. for @ptrCast: Zig guarantees that the address of a struct is the same as the address of its
            // first field. Since 'Vec' is a single-field struct containing '[N]T', its
            // in-memory representation is identical to the raw array. Therefore, casting the
            // address of the first element of an N-sized block to a '*Vec' is a valid,
            // zero-cost reinterpretation of the same memory.
            return @ptrCast(@alignCast(&self.data_buffer[index * N]));
        }
    };
}
