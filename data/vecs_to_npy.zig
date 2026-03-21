//! Convert fvecs, ivecs, and bvecs files to npy format.
//! fvecs -> f32, ivecs -> i32, bvecs -> u8
const std = @import("std");
const builtin = @import("builtin");
const znpy = @import("znpy");

const log = std.log.scoped(.vecs_to_npy);

pub const VecsType = enum {
    Fvecs,
    Ivecs,
    Bvecs,

    pub fn fromExtension(ext: []const u8) ?VecsType {
        if (std.mem.eql(u8, ext, ".fvecs")) return .Fvecs;
        if (std.mem.eql(u8, ext, ".ivecs")) return .Ivecs;
        if (std.mem.eql(u8, ext, ".bvecs")) return .Bvecs;
        return null;
    }
};

pub const VecsFileError = error{
    FileTooSmall,
    ReadFailed,
    VecsDimIsNegative,
    VecSizeOverflow,
    FileSizeNotMultipleOfVecSize,
    VecsDimExceedsUsize,
    NumVecsExceedsUsize,
    VecSizeExceedsUsize,
};

/// Read the first 4 bytes of the vecs file to get the vector dimension,
/// and check that the file size is consistent with the vector dimension to get the number of vectors.
/// Return in order:
/// 1. number of vectors (usize)
/// 2. vector dimension (usize)
fn getNumVecsAndVecsDim(
    vecs_type: VecsType,
    vecs_file: std.fs.File,
) (VecsFileError || std.fs.File.GetEndPosError)!struct { usize, usize } {
    var vecs_buffer: [4]u8 = undefined;
    var vecs_reader = vecs_file.reader(&vecs_buffer);
    const reader = &vecs_reader.interface;

    const vecs_dim_i32 = reader.takeInt(i32, .little) catch |err|
        return switch (err) {
            error.EndOfStream => error.FileTooSmall,
            error.ReadFailed => error.ReadFailed,
        };

    const vecs_file_size = try vecs_file.getEndPos();

    const vecs_dim_u64 = std.math.cast(u64, vecs_dim_i32) orelse return error.VecsDimIsNegative;
    const vec_size_u64 = std.math.add(
        u64,
        4,
        std.math.mul(
            u64,
            vecs_dim_u64,
            switch (vecs_type) {
                .Fvecs => 4,
                .Ivecs => 4,
                .Bvecs => 1,
            },
        ) catch return error.VecSizeOverflow,
    ) catch return error.VecSizeOverflow;

    const num_vecs_u64 = if (vecs_file_size % vec_size_u64 != 0) {
        return error.FileSizeNotMultipleOfVecSize;
    } else vecs_file_size / vec_size_u64;

    const num_vecs = std.math.cast(usize, num_vecs_u64) orelse return error.NumVecsExceedsUsize;
    const vecs_dim = std.math.cast(usize, vecs_dim_i32) orelse return error.VecsDimExceedsUsize;

    return .{ num_vecs, vecs_dim };
}

fn convert(
    vecs_type: VecsType,
    vecs_file: std.fs.File,
    npy_file: std.fs.File,
) (std.io.Reader.Error || std.io.Writer.Error || VecsFileError || std.fs.File.GetEndPosError)!void {
    const num_vecs, const vecs_dim = try getNumVecsAndVecsDim(vecs_type, vecs_file);
    var npy_buffer: [8192]u8 = undefined;
    var npy_writer = npy_file.writer(&npy_buffer);
    const writer = &npy_writer.interface;

    // Npy header for a 2D array of 2 usizes should not exceed 98 bytes, so 1024 bytes is more than enough for the header.
    var header_buffer: [1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&header_buffer);
    const allocator = fba.allocator();

    // Craft the header and write to npy file.
    const npy_header = znpy.header.Header{
        .descr = switch (vecs_type) {
            .Fvecs => .{ .Float32 = .little },
            .Ivecs => .{ .Int32 = .little },
            .Bvecs => .UInt8,
        },
        .order = .C,
        .shape = &[_]usize{ num_vecs, vecs_dim },
    };
    npy_header.writeAll(writer, allocator) catch |err| {
        switch (err) {
            error.HeaderTooLarge, error.OutOfMemory => @panic(
                "Header buffer is too small, should not happen since we allocated 1024 bytes for header which is more than enough." ++
                    " Znpy library is doing something unexpected if this happens.",
            ),
            error.WriteFailed => return std.io.Writer.Error.WriteFailed,
        }
    };

    var vecs_buffer: [8192]u8 = undefined;
    var vecs_reader = vecs_file.reader(&vecs_buffer);
    const reader = &vecs_reader.interface;

    var elem_buffer_scratch: [4]u8 = undefined; // big enough for one element of any type
    const elem_buffer = elem_buffer_scratch[0..switch (vecs_type) {
        .Fvecs, .Ivecs => 4,
        .Bvecs => 1,
    }];

    // Write one element at a time
    for (0..num_vecs) |_| {
        // We don't care if the dim value is different here
        // since we already checked the file size is consistent with the dim value.
        _ = try reader.takeInt(i32, .little);
        for (0..vecs_dim) |_| {
            try reader.readSliceAll(elem_buffer);
            if (builtin.cpu.arch.endian() != .little and vecs_type != .Bvecs) {
                // Need to swap endianness for multi-byte types if the CPU is not little-endian.
                std.mem.reverse(u8, elem_buffer);
            }
            try writer.writeAll(elem_buffer);
        }
    }

    // Flush the npy writer to ensure all data is written to the npy file.
    try writer.flush();
}

/// Convert all .fvecs, .ivecs, and .bvecs files in the given directory to .npy format.
pub fn convertVecsToNpy(dataset_dir: std.fs.Dir) void {
    var name_buffer_scratch: [std.fs.max_name_bytes]u8 = undefined;

    var iter = dataset_dir.iterate();
    while (iter.next() catch |err| {
        log.err("Error iterating directory, stopping: {}", .{err});
        return;
    }) |e| {
        const entry: std.fs.Dir.Entry = e;
        if (entry.kind != .file) continue;
        const ext = std.fs.path.extension(entry.name);
        const vecs_type = VecsType.fromExtension(ext) orelse continue;

        // Make new file name with .npy extension.
        const filename = entry.name[0 .. entry.name.len - ext.len];
        @memcpy(name_buffer_scratch[0..filename.len], filename);
        @memcpy(name_buffer_scratch[filename.len .. filename.len + ".npy".len], ".npy");
        const npy_name = name_buffer_scratch[0 .. filename.len + ".npy".len];

        // Open the vecs file for reading and the npy file for writing
        const vecs_file = dataset_dir.openFile(entry.name, .{}) catch |err| {
            log.err("Error opening vecs file {s} for reading, skipping: {}", .{ entry.name, err });
            continue;
        };
        defer vecs_file.close();
        const npy_file = dataset_dir.createFile(npy_name, .{}) catch |err| {
            log.err("Error creating npy file {s} for writing, skipping: {}", .{ npy_name, err });
            continue;
        };
        defer npy_file.close();

        convert(
            vecs_type,
            vecs_file,
            npy_file,
        ) catch |err| {
            switch (err) {
                error.EndOfStream => log.err("Unexpected end of vecs file {s}. File size has changed during conversion.", .{entry.name}),
                error.ReadFailed, error.WriteFailed => log.err("IO error while converting to npy for vecs file {s}", .{entry.name}),
                else => |conv_err| log.err("Error converting vecs file {s}: {}", .{ entry.name, conv_err }),
            }
            // Clear the npy file if we failed to convert, to avoid leaving a corrupted npy file.
            dataset_dir.deleteFile(npy_name) catch |delete_err|
                log.err("Error deleting npy file {s} after failed conversion, manual cleanup may be needed: {}", .{ npy_name, delete_err });
        };
        log.info("Successfully converted {s} to {s}", .{ entry.name, npy_name });
    }
}
