const std = @import("std");

/// A struct that represents a slice of data in a structure-of-arrays (SoA) format.
/// This struct owns the slice data and allocates memory for each field of the struct T separately,
/// allowing the slice of each field to have its own lifetime.
/// The slice cannot be resized after initialization.
pub fn SoaSlice(comptime T: type) type {
    switch (@typeInfo(T)) {
        .@"struct" => {},
        else => @compileError("T must be a struct type"),
    }
    const fields: []const std.builtin.Type.StructField = std.meta.fields(T);

    return struct {
        /// Byte pointers to the data for each field.
        ptrs: [fields.len][*]u8,
        /// Number of elements in the slice.
        len: usize,

        pub const Field = std.meta.FieldEnum(T);

        const Self = @This();

        /// Initializes a new SoaSlice with the specified length.
        pub fn init(len: usize, allocator: std.mem.Allocator) std.mem.Allocator.Error!Self {
            var ptrs: [fields.len][*]u8 = undefined;
            inline for (fields, 0..) |field, i| {
                const bytes = try allocator.alignedAlloc(
                    u8,
                    .of(field.type),
                    len * @sizeOf(field.type),
                );
                ptrs[i] = bytes.ptr;
            }
            return Self{
                .ptrs = ptrs,
                .len = len,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            inline for (0..fields.len) |i| {
                allocator.free(self.items(@enumFromInt(i)));
            }
        }

        /// Fills the slice with the values from the given element.
        pub fn fill(self: *Self, elem: T) void {
            inline for (fields, 0..) |field, i| {
                if (@sizeOf(field.type) == 0) continue; // Skip zero-sized fields. Otherwise LLVM cannot generate code for the pointer arithmetic.
                const field_enum = @as(Field, @enumFromInt(i));
                const field_value = @field(elem, field.name);
                @memset(self.items(field_enum), field_value);
            }
        }

        fn FieldType(comptime field: Field) type {
            return @FieldType(T, @tagName(field));
        }

        /// Returns a slice of the data for the specified field,
        /// allowing access to the individual field values for all elements in the slice.
        pub fn items(self: *const Self, comptime field: Field) []FieldType(field) {
            const F = FieldType(field);
            if (self.len == 0) return &[_]F{};
            const byte_slice = self.ptrs[@intFromEnum(field)];
            const casted_slice: [*]F = if (@sizeOf(F) == 0) undefined else @ptrCast(@alignCast(byte_slice));
            return casted_slice[0..self.len];
        }

        /// Set the element at the specified index to the given value.
        /// Index has to be less than the length of the slice.
        pub fn set(self: *Self, index: usize, elem: T) void {
            std.debug.assert(index < self.len);
            inline for (fields, 0..) |field, i| {
                self.items(@as(Field, @enumFromInt(i)))[index] = @field(elem, field.name);
            }
        }

        /// Get the element at the specified index.
        /// Index has to be less than the length of the slice.
        pub fn get(self: *const Self, index: usize) T {
            std.debug.assert(index < self.len);
            var elem: T = undefined;
            inline for (fields, 0..) |field, i| {
                @field(elem, field.name) = self.items(@as(Field, @enumFromInt(i)))[index];
            }
            return elem;
        }

        /// Returns a subslice of the original slice starting at the specified offset and with the specified length.
        /// off + len must be less than or equal to the length of the original slice.
        pub fn subslice(self: *const Self, off: usize, len: usize) Self {
            std.debug.assert(off + len <= self.len);
            var ptrs = self.ptrs;
            inline for (fields, 0..) |field, i| {
                ptrs[i] += off * @sizeOf(field.type);
            }
            return .{
                .ptrs = ptrs,
                .len = len,
            };
        }
    };
}

test "basic usage" {
    const allocator = std.testing.allocator;
    const testing = std.testing;

    const Foo = struct {
        a: u32,
        b: []const u8,
        c: u8,
    };

    var slice = try SoaSlice(Foo).init(3, allocator);
    defer slice.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), slice.items(.a).len);
    try testing.expectEqual(@as(usize, 3), slice.items(.b).len);
    try testing.expectEqual(@as(usize, 3), slice.items(.c).len);

    slice.set(0, .{ .a = 1, .b = "foobar", .c = 'a' });
    slice.set(1, .{ .a = 2, .b = "zigzag", .c = 'b' });
    slice.set(2, .{ .a = 3, .b = "fizzbuzz", .c = 'c' });

    try testing.expectEqualSlices(u32, slice.items(.a), &[_]u32{ 1, 2, 3 });
    try testing.expectEqualSlices(u8, slice.items(.c), &[_]u8{ 'a', 'b', 'c' });

    try testing.expectEqualStrings("foobar", slice.items(.b)[0]);
    try testing.expectEqualStrings("zigzag", slice.items(.b)[1]);
    try testing.expectEqualStrings("fizzbuzz", slice.items(.b)[2]);

    const subslice = slice.subslice(1, 2);
    try testing.expectEqual(@as(usize, 2), subslice.items(.a).len);
    try testing.expectEqualSlices(u32, subslice.items(.a), &[_]u32{ 2, 3 });
    try testing.expectEqualStrings("zigzag", subslice.items(.b)[0]);
    try testing.expectEqualStrings("fizzbuzz", subslice.items(.b)[1]);
    try testing.expectEqualSlices(u8, subslice.items(.c), &[_]u8{ 'b', 'c' });
}

test "struct with void field" {
    const allocator = std.testing.allocator;
    const testing = std.testing;

    const Bar = struct {
        x: u32,
        y: void,
    };

    var slice = try SoaSlice(Bar).init(2, allocator);
    defer slice.deinit(allocator);

    slice.set(0, .{ .x = 42, .y = {} });
    slice.set(1, .{ .x = 99, .y = {} });

    try testing.expectEqualSlices(u32, slice.items(.x), &[_]u32{ 42, 99 });
    try testing.expectEqualSlices(void, slice.items(.y), &[_]void{ {}, {} });
}
