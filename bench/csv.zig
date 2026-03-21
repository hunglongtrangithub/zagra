const std = @import("std");
const config = @import("config");

/// Directory to store csv files from running benchmarks
pub const CSV_RESULTS_DIR: []const u8 = config.BENCH_DIR ++ "/results";

const Error = error{RowLengthMismatch} || std.Io.Writer.Error;

pub fn writeCsv(
    writer: *std.Io.Writer,
    headers: []const []const u8,
    rows: anytype,
) Error!void {
    try writeHeaders(writer, headers);
    for (rows) |row| {
        try writeRow(writer, row, headers.len);
    }
}

pub fn writeRow(writer: *std.Io.Writer, row: anytype, expected_count: usize) Error!void {
    try writeRowValues(writer, row, expected_count);
    try writer.writeAll("\n");
}

pub fn writeHeaders(
    writer: *std.Io.Writer,
    headers: []const []const u8,
) std.Io.Writer.Error!void {
    if (headers.len > 0) {
        try writeEscaped(writer, headers[0]);
        for (headers[1..]) |value| {
            try writer.writeAll(",");
            try writeEscaped(writer, value);
        }
    }
    try writer.writeAll("\n");
}

fn writeRowValues(
    writer: *std.Io.Writer,
    row: anytype,
    expected_count: usize,
) Error!void {
    const T = @TypeOf(row);
    switch (@typeInfo(T)) {
        .array => {
            if (row.len != expected_count) return error.RowLengthMismatch;
            if (row.len > 0) {
                try writeCsvValue(writer, row[0]);
                for (row[1..]) |value| {
                    try writer.writeAll(",");
                    try writeCsvValue(writer, value);
                }
            }
        },
        .@"struct" => {
            const fields = std.meta.fields(T);
            if (fields.len != expected_count) return error.RowLengthMismatch;
            if (fields.len > 0) {
                try writeCsvValue(writer, @field(row, fields[0].name));
                inline for (fields[1..]) |field| {
                    try writer.writeAll(",");
                    try writeCsvValue(writer, @field(row, field.name));
                }
            }
        },
        .pointer => |ptr| switch (ptr.size) {
            .slice => {
                if (row.len != expected_count) return error.RowLengthMismatch;
                if (row.len > 0) {
                    try writeCsvValue(writer, row[0]);
                    for (row[1..]) |value| {
                        try writer.writeAll(",");
                        try writeCsvValue(writer, value);
                    }
                }
            },
            .one => try writeRowValues(writer, row.*, expected_count),
            else => @compileError("unsupported pointer size: " ++ @typeName(T)),
        },
        else => @compileError("unsupported row type: " ++ @typeName(T)),
    }
}

fn writeCsvValue(writer: *std.Io.Writer, value: anytype) std.Io.Writer.Error!void {
    const T = @TypeOf(value);
    switch (@typeInfo(T)) {
        .pointer => |ptr| switch (ptr.size) {
            .slice => if (ptr.child == u8) {
                try writeEscaped(writer, value);
            } else {
                @compileError("unsupported slice type: " ++ @typeName(T));
            },
            .one => try writeCsvValue(writer, value.*),
            else => @compileError("unsupported pointer type: " ++ @typeName(T)),
        },
        .array => |arr| if (arr.child == u8) {
            try writeEscaped(writer, &value);
        } else {
            @compileError("unsupported array type: " ++ @typeName(T));
        },
        .int, .comptime_int => try writer.print("{d}", .{value}),
        .float, .comptime_float => try writer.print("{d}", .{value}),
        .bool => try writer.writeAll(if (value) "true" else "false"),
        .optional => if (value) |v| try writeCsvValue(writer, v) else try writer.writeAll(""),
        else => @compileError("unsupported type: " ++ @typeName(T)),
    }
}

fn writeEscaped(writer: *std.Io.Writer, value: []const u8) std.Io.Writer.Error!void {
    const needs_escape = std.mem.indexOfAny(u8, value, ",\"\n\r") != null;
    if (needs_escape) {
        try writer.writeAll("\"");
        var escaped = value;
        while (std.mem.indexOfScalar(u8, escaped, '"')) |idx| {
            try writer.writeAll(escaped[0..idx]);
            try writer.writeAll("\"\"");
            escaped = escaped[idx + 1 ..];
        }
        try writer.writeAll(escaped);
        try writer.writeAll("\"");
    } else {
        try writer.writeAll(value);
    }
}

test "writeRow empty" {
    var buf: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try writeRow(&writer, &[_][]const u8{}, 0);
    try std.testing.expectEqualStrings("\n", writer.buffered());
}

test "writeRow single value" {
    var buf: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try writeRow(&writer, &[_][]const u8{"hello"}, 1);
    try std.testing.expectEqualStrings("hello\n", writer.buffered());
}

test "writeRow multiple values" {
    var buf: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try writeRow(&writer, &[_][]const u8{ "hello", "world", "foo" }, 3);
    try std.testing.expectEqualStrings("hello,world,foo\n", writer.buffered());
}

test "writeEscaped comma" {
    var buf: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try writeRow(&writer, &[_][]const u8{"hello,world"}, 1);
    try std.testing.expectEqualStrings("\"hello,world\"\n", writer.buffered());
}

test "writeEscaped quote" {
    var buf: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try writeRow(&writer, &[_][]const u8{"hello\"world"}, 1);
    try std.testing.expectEqualStrings("\"hello\"\"world\"\n", writer.buffered());
}

test "writeEscaped newline" {
    var buf: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try writeRow(&writer, &[_][]const u8{"hello\nworld"}, 1);
    try std.testing.expectEqualStrings("\"hello\nworld\"\n", writer.buffered());
}

test "writeEscaped no escaping needed" {
    var buf: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try writeRow(&writer, &[_][]const u8{"hello world"}, 1);
    try std.testing.expectEqualStrings("hello world\n", writer.buffered());
}

test "writeCsv strings" {
    var buf: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try writeCsv(
        &writer,
        &[_][]const u8{ "name", "city" },
        &[_][2][]const u8{
            .{ "Alice", "New York" },
            .{ "Bob", "San, Francisco" },
        },
    );
    try std.testing.expectEqualStrings(
        "name,city\nAlice,New York\nBob,\"San, Francisco\"\n",
        writer.buffered(),
    );
}

test "writeCsv mixed types" {
    var buf: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try writeCsv(
        &writer,
        &[_][]const u8{ "name", "age", "score", "active" },
        &[_]struct { []const u8, u32, f32, bool }{
            .{ "Alice", 30, 9.5, true },
            .{ "Bob", 25, 7.2, false },
        },
    );
    try std.testing.expectEqualStrings(
        "name,age,score,active\nAlice,30,9.5,true\nBob,25,7.2,false\n",
        writer.buffered(),
    );
}

test "writeCsv optionals" {
    var buf: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try writeCsv(
        &writer,
        &[_][]const u8{ "name", "nickname" },
        &[_]struct { []const u8, ?[]const u8 }{
            .{ "Alice", "Ali" },
            .{ "Bob", null },
        },
    );
    try std.testing.expectEqualStrings(
        "name,nickname\nAlice,Ali\nBob,\n",
        writer.buffered(),
    );
}

test "writeRow length mismatch" {
    var buf: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    // Provide 2 values but expected_count is 1 -> should return RowLengthMismatch
    try std.testing.expectError(Error.RowLengthMismatch, writeRow(&writer, &[_][]const u8{ "a", "b" }, 1));
}
