pub const ElemType = enum {
    // Int8,
    // UInt8,
    Int32,
    Float,
    Half,

    const Self = @This();

    pub fn fromZigType(comptime T: type) ?Self {
        return switch (T) {
            // i8 => Self.Int8,
            // u8 => Self.UInt8,
            i32 => Self.Int32,
            f32 => Self.Float,
            f16 => Self.Half,
            else => null,
        };
    }

    pub fn toZigType(self: Self) type {
        return switch (self) {
            // .Int8 => i8,
            // .UInt8 => u8,
            .Int32 => i32,
            .Float => f32,
            .Half => f16,
        };
    }
};

pub const DimType = enum {
    D128,
    D256,
    D512,

    const Self = @This();

    pub fn fromDim(n: usize) ?Self {
        return switch (n) {
            128 => Self.D128,
            256 => Self.D256,
            512 => Self.D512,
            else => null,
        };
    }

    pub fn toDim(self: Self) usize {
        return switch (self) {
            .D128 => 128,
            .D256 => 256,
            .D512 => 512,
        };
    }
};
