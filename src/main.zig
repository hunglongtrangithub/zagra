const std = @import("std");

const znpy = @import("znpy");
const zagra = @import("zagra");

var stdout_buffer: [1024]u8 = undefined;
var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
const stdout = &stdout_writer.interface;

pub fn main() !void {
    std.debug.print("This is Zagra!\n", .{});

    // Dataset configuration constants
    const vector_length: usize = 128;
    const vector_count: usize = 500;
    const T = f32;
    const npy_file_name = "dataset.npy";
    try stdout.print("Npy array dimensions: {} x {}\n", .{ vector_count, vector_length });

    // Open file to write Npy array to
    const npy_file_to_write = try std.fs.cwd().createFile(npy_file_name, .{});
    var file_buffer: [8192]u8 = undefined;
    var file_writer = std.fs.File.Writer.init(npy_file_to_write, &file_buffer);

    const allocator = std.heap.page_allocator;

    // Initialize the Npy array
    const Array = znpy.array.StaticArray(T, 2);
    const array = try Array.init(
        [2]usize{ vector_count, vector_length },
        znpy.shape.Order.C,
        allocator,
    );
    defer array.deinit(allocator);
    // Fill array with increasing numbers
    for (0..vector_count) |i| {
        for (0..vector_length) |j| {
            array.at([2]usize{ i, j }).?.* = @floatFromInt(i * vector_length + j);
        }
    }
    // Write file to disk
    try array.writeAll(&file_writer.interface, allocator);
    npy_file_to_write.close();

    try stdout.print("Wrote Npy array to {s}\n", .{npy_file_name});
    try stdout.flush();

    try stdout.print("Reading back the Npy file to dataset\n", .{});

    // Read the same file back
    const npy_file_to_read = try std.fs.cwd().openFile(npy_file_name, .{});
    defer npy_file_to_read.close();
    // Reuse the file buffer for the file reader
    var file_reader = std.fs.File.Reader.init(npy_file_to_read, &file_buffer);

    // Read the Npy file content into the dataset
    const Dataset = zagra.dataset.Dataset(T, vector_length);
    const dataset = try Dataset.fromNpyFileReader(&file_reader.interface, allocator);
    defer dataset.deinit(allocator);

    try stdout.print("First two vectors of the dataset:\n", .{});
    for (0..2) |i| {
        const vector = dataset.getUnchecked(i);
        try stdout.print("Vector {}: {any}\n", .{ i, vector.*.data });
    }
    try stdout.flush();
}
