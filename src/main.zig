const std = @import("std");

const znpy = @import("znpy");
const zagra = @import("zagra");

pub const std_options: std.Options = .{
    .log_level = .err,
};

var stdout_buffer: [1024]u8 = undefined;
var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
const stdout = &stdout_writer.interface;

const HELP =
    \\zagra <vector_count> <block_processing>
    \\- vector_count (required): Number of vectors in the dataset
    \\- block_processing (optional - default to true): Whether to use block processing mode during training
;

pub fn main() !void {
    std.debug.print("This is Zagra!\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);

    _ = args.skip();

    const vector_count_str = args.next() orelse {
        std.debug.print("vector count needed.{s}\n", .{HELP});
        return;
    };
    const vector_count = std.fmt.parseInt(usize, vector_count_str, 10) catch |e| {
        switch (e) {
            error.Overflow => std.debug.print("Entered input is too large for usize\n", .{}),
            error.InvalidCharacter => std.debug.print("Entered input is not a valid number\n", .{}),
        }
        return;
    };

    const block_processing = blk: {
        if (args.next()) |str| {
            if (std.mem.eql(u8, str, "true"))
                break :blk true
            else if (std.mem.eql(u8, str, "false"))
                break :blk false
            else {
                std.debug.print("Expected 'true' or 'false' for block_processing", .{});
                return;
            }
        } else break :blk true;
    };
    try stdout.print("Using block processing for training? {any}\n", .{block_processing});

    // Dataset configuration constants
    const vector_length: usize = 128;
    const npy_file_name = "dataset.npy";
    const element_type = f32;

    try stdout.print("Saving dataset with {} {}-D vectors to {s}\n", .{ vector_count, vector_length, npy_file_name });
    try stdout.flush();

    // Open file to write Npy array to
    const npy_file_to_write = try std.fs.cwd().createFile(npy_file_name, .{});
    var file_buffer: [8192]u8 = undefined;
    var file_writer = std.fs.File.Writer.init(npy_file_to_write, &file_buffer);

    // Initialize the Npy array
    const Array = znpy.array.StaticArray(element_type, 2);
    const array = Array.init(
        [2]usize{ vector_count, vector_length },
        znpy.shape.Order.C,
        allocator,
    ) catch |e| {
        std.debug.print("Error initializing array: {}", .{e});
        return;
    };
    defer array.deinit(allocator);
    // Fill array with increasing numbers
    for (0..vector_count) |i| {
        for (0..vector_length) |j| {
            array.at([2]usize{ i, j }).?.* = @floatFromInt(i * vector_length + j);
        }
    }
    // Write file to disk
    array.writeAll(&file_writer.interface, allocator) catch |e| {
        std.debug.print("Error writing array to disk: {}", .{e});
        return;
    };
    npy_file_to_write.close();
    try stdout.print("Wrote Npy array to {s}\n", .{npy_file_name});
    try stdout.flush();

    try stdout.print("Reading the Npy file to dataset\n", .{});

    const npy_file_to_read = try std.fs.cwd().openFile(npy_file_name, .{});
    const Dataset = zagra.dataset.Dataset(element_type, vector_length);

    // Read the file
    // var file_buffer: [8192]u8 = undefined;
    var file_reader = std.fs.File.Reader.init(npy_file_to_read, &file_buffer);

    // Read the Npy file content into the dataset
    const dataset = Dataset.fromNpyFileReader(&file_reader.interface, allocator) catch |e| {
        std.debug.print("Error creating the dataset from the array file: {}", .{e});
        return;
    };
    defer dataset.deinit(allocator);

    // // Get the file buffer using mmap
    //
    // // Get file size
    // const file_stat = try npy_file_to_read.stat();
    // const read_size = std.math.cast(usize, file_stat.size) orelse {
    //     std.debug.print("File size is too large to map\n", .{});
    //     return;
    // };
    // if (read_size == 0) {
    //     std.debug.print("File is empty, nothing to read\n", .{});
    //     return;
    // }
    //
    // // Read all file contents into memory using mmap
    // const file_buffer = try std.posix.mmap(
    //     null,
    //     read_size,
    //     std.posix.PROT.READ,
    //     std.posix.system.MAP{ .TYPE = .PRIVATE },
    //     npy_file_to_read.handle,
    //     0,
    // );
    // defer std.posix.munmap(file_buffer);
    // if (file_buffer.len != read_size) {
    //     std.debug.print("Mapped size does not match file size.\n", .{});
    //     return;
    // }
    //
    // // Form the dataset using the file buffer
    // const dataset = try Dataset.fromNpyFileBuffer(file_buffer, allocator);

    std.debug.assert(dataset.len == vector_count);

    npy_file_to_read.close();

    // for (0..dataset.len) |i| {
    //     const vector = dataset.getUnchecked(i);
    //     try stdout.print("vector {}: {any}\n", .{ i, vector.data });
    // }
    // try stdout.flush();

    // Do NN-Descent
    const NNDescent = zagra.graphs.nn_descent.NNDescent(element_type, vector_length);
    var training_config = zagra.graphs.nn_descent.TrainingConfig.init(
        4,
        vector_count,
        null,
        42,
    );
    training_config.block_processing = block_processing;
    var nn_descent = NNDescent.init(
        dataset,
        training_config,
        allocator,
    ) catch |e| {
        std.debug.print("Error initializing NNDescent: {}", .{e});
        return;
    };
    defer nn_descent.deinit(allocator);

    try stdout.print("Start timing NN-Descent training...\n", .{});
    try stdout.flush();

    var timer = try std.time.Timer.start();
    nn_descent.train();
    const elapsed_time_ns = timer.read();
    const elapsed_time_s: f64 = @as(f64, @floatFromInt(elapsed_time_ns)) / @as(f64, @floatFromInt(1_000_000_000));

    try stdout.print("Training for {} vectors took: {}s\n", .{ dataset.len, elapsed_time_s });
    try stdout.flush();

    // const neighbors_list = nn_descent.neighbors_list;
    //
    // try stdout.print("neighbors_list: {} nodes x {} neighbors per node\n", .{ neighbors_list.num_nodes, neighbors_list.num_neighbors_per_node });
    // std.debug.assert(neighbors_list.num_nodes * neighbors_list.num_neighbors_per_node == neighbors_list.entries.len);
    //
    // for (0..neighbors_list.entries.len) |i| {
    //     try stdout.print("neighbors_list entry at index {}: {any}\n", .{ i, neighbors_list.entries.get(i) });
    // }
}
