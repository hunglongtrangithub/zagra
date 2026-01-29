const std = @import("std");

const znpy = @import("znpy");
const zagra = @import("zagra");

var stdout_buffer: [1024]u8 = undefined;
var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
const stdout = &stdout_writer.interface;

fn saveDataset(
    comptime element_type: type,
    vector_length: usize,
    vector_count: usize,
    npy_sub_path: []const u8,
    allocator: std.mem.Allocator,
) !void {
    // Open file to write Npy array to
    const npy_file_to_write = try std.fs.cwd().createFile(npy_sub_path, .{});
    var file_buffer: [8192]u8 = undefined;
    var file_writer = std.fs.File.Writer.init(npy_file_to_write, &file_buffer);

    // Initialize the Npy array
    const Array = znpy.array.StaticArray(element_type, 2);
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
}

pub fn main() !void {
    std.debug.print("This is Zagra!\n", .{});

    // Dataset configuration constants
    const vector_length: usize = 128;
    const vector_count: usize = 1000000;
    const npy_file_name = "dataset.npy";
    const element_type = f32;
    var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
    const allocator = gpa.allocator();

    try saveDataset(
        element_type,
        vector_length,
        vector_count,
        npy_file_name,
        allocator,
    );
    try stdout.print("Wrote Npy array to {s}\n", .{npy_file_name});
    try stdout.flush();

    try stdout.print("Reading the Npy file to dataset\n", .{});

    // Read the file
    const npy_file_to_read = try std.fs.cwd().openFile(npy_file_name, .{});
    var file_buffer: [8192]u8 = undefined;
    var file_reader = std.fs.File.Reader.init(npy_file_to_read, &file_buffer);

    // Read the Npy file content into the dataset
    const Dataset = zagra.dataset.Dataset(element_type, vector_length);
    const dataset = try Dataset.fromNpyFileReader(&file_reader.interface, allocator);
    defer dataset.deinit(allocator);

    std.debug.assert(dataset.len == vector_count);

    npy_file_to_read.close();

    // for (0..dataset.len) |i| {
    //     const vector = dataset.getUnchecked(i);
    //     try stdout.print("vector {}: {any}\n", .{ i, vector.data });
    // }
    // try stdout.flush();

    // Do NN-Descent
    const NNDescent = zagra.graphs.nn_descent.NNDescent(element_type, vector_length);
    const training_config = zagra.graphs.nn_descent.TrainingConfig.init(
        4,
        vector_count,
        1,
        42,
    );
    var nn_descent = NNDescent.init(
        dataset,
        training_config,
        allocator,
    ) catch |e| {
        std.debug.print("Error initializing NNDescent: {}", .{e});
        return;
    };
    defer nn_descent.deinit(allocator);

    nn_descent.train();

    // const neighbors_list = nn_descent.neighbors_list;
    //
    // try stdout.print("neighbors_list: {} nodes x {} neighbors per node\n", .{ neighbors_list.num_nodes, neighbors_list.num_neighbors_per_node });
    // std.debug.assert(neighbors_list.num_nodes * neighbors_list.num_neighbors_per_node == neighbors_list.entries.len);
    //
    // for (0..neighbors_list.entries.len) |i| {
    //     try stdout.print("neighbors_list entry at index {}: {any}\n", .{ i, neighbors_list.entries.get(i) });
    // }
    // try stdout.flush();
}
