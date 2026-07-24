//! Simple FTP client for downloading files
//! Supports anonymous login and passive mode transfers

const std = @import("std");
const builtin = @import("builtin");

pub const FtpError = error{
    InvalidUrl,
    ConnectionFailed,
    LoginFailed,
    CommandFailed,
    FileNotFound,
    TransferFailed,
    FileCreateFailed,
};

pub const DownloadResult = FtpError!u64;

/// A file to download
pub const DownloadItem = struct {
    /// Full URL to download (e.g., "ftp://example.com/path/to/file.tar.gz")
    url: []const u8,
    /// Path to save the downloaded file. Can be either:
    /// - Absolute path (e.g., "/tmp/file.tar.gz")
    /// - Relative path to current working directory (e.g., "data/file.tar.gz")
    output_path: []const u8,
};

/// Downloads multiple files in parallel with progress display.
/// Returns slice of results for each file, in the same order as input items.
/// `items` must have <= 255 entries, otherwise `error.TooManyFiles` is returned.
pub fn downloadFiles(io: std.Io, allocator: std.mem.Allocator, items: []const DownloadItem) error{ TooManyFiles, OutOfMemory }![]DownloadResult {
    if (items.len == 0) return &.{};

    const total_tasks = std.math.cast(u8, items.len) orelse return error.TooManyFiles;

    // Allocate results and contexts
    const contexts = try allocator.alloc(DownloadContext, items.len);
    defer allocator.free(contexts);
    const results = try allocator.alloc(DownloadResult, items.len);

    // Initialize contexts
    for (items, 0..) |item, i| {
        results[i] = error.ConnectionFailed; // Default to error, will be overwritten
        contexts[i] = .{
            .io = io,
            .allocator = allocator,
            .url = item.url,
            .output_path = item.output_path,
            .filename = getFilename(item.url),
            .task_id = @intCast(i),
            .task_count = total_tasks,
            .result = &results[i],
        };
    }

    // Reserve lines for progress bars
    for (0..total_tasks) |_| std.debug.print("\n", .{});

    // Hide cursor during downloads
    std.debug.print("\x1b[?25l", .{});
    defer std.debug.print("\x1b[?25h\n", .{}); // Show cursor when done

    var group = std.Io.Group.init;
    defer group.cancel(io);
    for (contexts) |*ctx| {
        group.async(io, DownloadContext.run, .{ctx});
    }
    // Wait for all successfully spawned tasks.
    // Download errors as stored in results already.
    group.await(io) catch {};
    return results;
}

fn getFilename(url: []const u8) []const u8 {
    // Find last '/' and return everything after it
    if (std.mem.lastIndexOf(u8, url, "/")) |idx| {
        return url[idx + 1 ..];
    }
    return url;
}

test getFilename {
    try std.testing.expectEqualStrings("file.tar.gz", getFilename("ftp://example.com/path/to/file.tar.gz"));
    try std.testing.expectEqualStrings("file.txt", getFilename("/just/a/path/file.txt"));
    try std.testing.expectEqualStrings("nopath", getFilename("nopath"));
}

/// Download context containing all state for a single download task
const DownloadContext = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    url: []const u8,
    output_path: []const u8,
    filename: []const u8,
    task_id: u8,
    task_count: u8,
    result: *DownloadResult,

    const DEFAULT_PORT: u16 = 21;

    const Response = struct {
        code: u16,
        /// owned by caller
        message: []const u8,
    };

    const Status = union(enum) {
        info,
        err,
        progress: struct { downloaded: u64, total: u64 },
        done: u64,
    };

    const Self = @This();

    /// Thread entry point
    fn run(self: *Self) void {
        self.result.* = self.download();
    }

    fn greeting(self: *Self, reader: *std.Io.Reader, arena: std.mem.Allocator) error{ConnectionFailed}!void {
        const welcome = readResponse(reader, arena) catch |e| {
            self.printStatus(.err, "{}: Failed to receive server's welcome message", .{e});
            return error.ConnectionFailed;
        };
        if (welcome.code != 220) {
            self.printStatus(.err, "Server is not ready (code: {})", .{welcome.code});
            return error.ConnectionFailed;
        }
    }

    /// The "USER" and "PASS" commands for login. Uses anonymous login with a dummy password if no credentials are provided.
    fn login(
        self: *Self,
        reader: *std.Io.Reader,
        writer: *std.Io.Writer,
        arena: std.mem.Allocator,
    ) error{ LoginFailed, ConnectionFailed }!void {
        // USER (use "anonymous" for anonymous login)
        sendCommand(writer, "USER", &.{"anonymous"}) catch |e| {
            self.printStatus(.err, "{}: Failed to send USER command", .{e});
            return error.ConnectionFailed;
        };
        const user_resp = readResponse(reader, arena) catch |e| {
            self.printStatus(.err, "{}: Failed to read USER response", .{e});
            return error.ConnectionFailed;
        };

        // 331: User name okay, need password.
        // 230: User logged in, proceed. (In this case we can skip PASS)
        if (user_resp.code != 331 and user_resp.code != 230) {
            self.printStatus(.err, "USER command failed (code: {})", .{user_resp.code});
            return error.LoginFailed;
        }

        // PASS (use dummy password, same as wget's fallback password)
        if (user_resp.code == 331) {
            sendCommand(writer, "PASS", &.{"-wget@"}) catch |e| {
                self.printStatus(.err, "{}: Failed to send PASS command", .{e});
                return error.ConnectionFailed;
            };
            const pass_resp = readResponse(reader, arena) catch |e| {
                self.printStatus(.err, "{}: Failed to read PASS response", .{e});
                return error.ConnectionFailed;
            };

            if (pass_resp.code != 230) {
                self.printStatus(.err, "Login failed (code: {})", .{pass_resp.code});
                return error.LoginFailed;
            }
        }
    }

    fn syst(
        self: *Self,
        reader: *std.Io.Reader,
        writer: *std.Io.Writer,
        allocator: std.mem.Allocator,
    ) error{ CommandFailed, ConnectionFailed }![]const u8 {
        sendCommand(writer, "SYST", &.{}) catch |e| {
            self.printStatus(.err, "{}: Failed to send SYST command", .{e});
            return error.ConnectionFailed;
        };
        const syst_resp = readResponse(reader, allocator) catch |e| {
            self.printStatus(.err, "{}: Failed to read SYST response", .{e});
            return error.ConnectionFailed;
        };
        if (syst_resp.code != 215) {
            self.printStatus(.err, "SYST command failed (code: {})", .{syst_resp.code});
            return error.CommandFailed;
        }
        return syst_resp.message;
    }

    /// The "TYPE" command. Sets transfer type from the query parameter "type" in the URL, or defaults to binary ("I").
    fn set_transfer_type(
        self: *Self,
        query: ?[]const u8,
        reader: *std.Io.Reader,
        writer: *std.Io.Writer,
        arena: std.mem.Allocator,
    ) error{ CommandFailed, ConnectionFailed }!void {
        const transfer_type = if (query != null and std.mem.startsWith(u8, query.?, "type="))
            std.ascii.toUpper(query.?[5])
        else
            'I';
        sendCommand(writer, "TYPE", &.{&.{transfer_type}}) catch |e| {
            self.printStatus(.err, "{}: Failed to send TYPE command", .{e});
            return error.ConnectionFailed;
        };
        const type_resp = readResponse(reader, arena) catch |e| {
            self.printStatus(.err, "{}: Failed to read TYPE response", .{e});
            return error.ConnectionFailed;
        };
        if (type_resp.code != 200) {
            self.printStatus(.err, "TYPE command failed (code: {})", .{type_resp.code});
            return error.CommandFailed;
        }
    }

    /// The "SIZE" command. Returns the file size in bytes, or null if any error occurs.
    fn get_file_size(
        self: *Self,
        file_path: []const u8,
        reader: *std.Io.Reader,
        writer: *std.Io.Writer,
        arena: std.mem.Allocator,
    ) ?u64 {
        // Use the SIZE command to get the file size before downloading. Fall back to 0 if any error occurs.
        sendCommand(writer, "SIZE", &.{file_path}) catch |e| {
            self.printStatus(.err, "{}: Failed to send SIZE command", .{e});
            return null;
        };
        const size_resp = readResponse(reader, arena) catch |e| {
            self.printStatus(.err, "{}: Failed to read SIZE response", .{e});
            return null;
        };
        if (size_resp.code != 213) {
            self.printStatus(.err, "SIZE command failed (code: {})", .{size_resp.code});
            return null;
        }
        return std.fmt.parseInt(u64, size_resp.message, 10) catch |e| {
            self.printStatus(.err, "{}: Failed to parse file size", .{e});
            return null;
        };
    }

    fn retr(
        self: *Self,
        writer: *std.Io.Writer,
        reader: *std.Io.Reader,
        file_path: []const u8,
        arena: std.mem.Allocator,
    ) error{ ConnectionFailed, CommandFailed, FileNotFound }!void {
        sendCommand(writer, "RETR", &.{file_path}) catch |e| {
            self.printStatus(.err, "{}: Failed to send RETR command", .{e});
            return error.ConnectionFailed;
        };
        const retr_resp = readResponse(reader, arena) catch |e| {
            self.printStatus(.err, "{}: Failed to read RETR response", .{e});
            return error.ConnectionFailed;
        };
        if (retr_resp.code == 550) {
            self.printStatus(.err, "File not found (code: {})", .{retr_resp.code});
            return error.FileNotFound;
        }
        if (retr_resp.code != 150) {
            self.printStatus(.err, "RETR command failed (code: {})", .{retr_resp.code});
            return error.CommandFailed;
        }
    }

    fn pasv(
        self: *Self,
        reader: *std.Io.Reader,
        writer: *std.Io.Writer,
        arena: std.mem.Allocator,
    ) error{ ConnectionFailed, InvalidPasvResponse }!std.Io.net.Ip4Address {
        sendCommand(writer, "PASV", &.{}) catch |e| {
            self.printStatus(.err, "{}: Failed to send PASV command", .{e});
            return error.ConnectionFailed;
        };
        const pasv_resp = readResponse(reader, arena) catch |e| {
            self.printStatus(.err, "{}: Failed to read PASV response", .{e});
            return error.ConnectionFailed;
        };
        if (pasv_resp.code != 227) {
            self.printStatus(.err, "PASV command failed (code: {})", .{pasv_resp.code});
            return error.ConnectionFailed;
        }
        const addr = parsePasvResponse(pasv_resp.message) catch |e| {
            self.printStatus(.err, "{}: Failed to parse PASV response", .{e});
            return error.InvalidPasvResponse;
        };
        return addr;
    }

    const ParsePasvError = error{
        MissingParenthesis,
        MisplacedParentheses,
        TooManyParts,
        NotEnoughParts,
        ParseIntFailed,
    };
    /// Parse PASV response: "Entering Passive Mode (h1,h2,h3,h4,p1,p2)"
    fn parsePasvResponse(msg: []const u8) ParsePasvError!std.Io.net.Ip4Address {
        const start = std.mem.find(u8, msg, "(") orelse return ParsePasvError.MissingParenthesis;
        const end = std.mem.find(u8, msg, ")") orelse return ParsePasvError.MissingParenthesis;
        if (start >= end) return ParsePasvError.MisplacedParentheses;

        const nums_str = msg[start + 1 .. end];

        // Parse 6 comma-separated numbers
        var nums: [6]u8 = undefined;
        var iter = std.mem.splitScalar(u8, nums_str, ',');
        var i: usize = 0;
        while (iter.next()) |part| {
            if (i >= 6) return ParsePasvError.TooManyParts;
            nums[i] = std.fmt.parseInt(u8, std.mem.trim(u8, part, " "), 10) catch return ParsePasvError.ParseIntFailed;
            i += 1;
        }
        if (i != 6) return ParsePasvError.NotEnoughParts;

        // Port = p1 * 256 + p2
        const port: u16 = @as(u16, nums[4]) * 256 + nums[5];

        return std.Io.net.Ip4Address{ .bytes = .{ nums[0], nums[1], nums[2], nums[3] }, .port = port };
    }
    test parsePasvResponse {
        const addr = try parsePasvResponse("Entering Passive Mode (131,254,14,19,196,108)");
        const expected = std.net.Address{ .in = std.net.Ip4Address.init(.{ 131, 254, 14, 19 }, 196 * 256 + 108) };
        try std.testing.expectEqual(expected, addr);
    }

    fn epsv(
        self: *const Self,
        control_addr: std.Io.net.IpAddress,
        reader: *std.Io.Reader,
        writer: *std.Io.Writer,
        arena: std.mem.Allocator,
    ) error{ ConnectionFailed, CommandFailed }!u16 {
        const addr_family = switch (control_addr) {
            .ip4 => "1",
            .ip6 => "2",
        };
        sendCommand(writer, "EPSV", &.{addr_family}) catch |e| {
            self.printStatus(.err, "{}: Failed to send EPSV command", .{e});
            return error.ConnectionFailed;
        };
        const epsv_resp = readResponse(reader, arena) catch |e| {
            self.printStatus(.err, "{}: Failed to read EPSV response", .{e});
            return error.ConnectionFailed;
        };
        if (epsv_resp.code != 229) {
            self.printStatus(.err, "EPSV command failed (code: {})", .{epsv_resp.code});
            return error.CommandFailed;
        }
        const data_port = parseEpsvResponse(epsv_resp.message) catch |e| {
            self.printStatus(.err, "{}: Failed to parse EPSV response", .{e});
            return error.CommandFailed;
        };
        return data_port;
    }

    const ParseEpsvError = error{
        MissingParenthesis,
        MisplacedParentheses,
        NotEnoughParts,
        PortParseFailed,
    };
    fn parseEpsvResponse(msg: []const u8) ParseEpsvError!u16 {
        // EPSV response format is "Entering Extended Passive Mode (<d><d><d>port<d>)"
        const start = std.mem.find(u8, msg, "(") orelse return ParseEpsvError.MissingParenthesis;
        const end = std.mem.find(u8, msg, ")") orelse return ParseEpsvError.MissingParenthesis;
        if (start >= end) return ParseEpsvError.MisplacedParentheses;

        const content = msg[start + 1 .. end];
        if (content.len < 4) return ParseEpsvError.NotEnoughParts;
        const delim = content[0];

        var iter = std.mem.splitScalar(u8, content, delim);
        const first_part = iter.next() orelse return ParseEpsvError.PortParseFailed;
        if (first_part.len != 0) return ParseEpsvError.PortParseFailed;
        const second_part = iter.next() orelse return ParseEpsvError.PortParseFailed;
        if (second_part.len != 0) return ParseEpsvError.PortParseFailed;
        const port_part = iter.next() orelse return ParseEpsvError.PortParseFailed;
        const port = std.fmt.parseInt(u16, port_part, 10) catch return ParseEpsvError.PortParseFailed;
        return port;
    }
    test parseEpsvResponse {
        const port = try parseEpsvResponse("Entering Extended Passive Mode (|||6446|)");
        try std.testing.expectEqual(6446, port);
    }

    /// Start the file transfer by connecting to the data address provided by PASV/EPSV response and sending the RETR command.
    /// Returns the data stream if successful, which the caller can read from to get the file contents.
    fn start_file_transfer(
        self: *Self,
        control_addr: std.Io.net.IpAddress,
        file_path: []const u8,
        reader: *std.Io.Reader,
        writer: *std.Io.Writer,
        arena: std.mem.Allocator,
    ) error{ ConnectionFailed, CommandFailed, FileNotFound }!std.Io.net.Stream {
        // from wget source code:
        // "If our control connection is over IPv6, then we first try EPSV and then
        // LPSV if the former is not supported. If the control connection is over
        // IPv4, we simply issue the good old PASV request."
        self.printStatus(.info, "Connecting to data address...", .{});
        const data_stream = ds: switch (control_addr) {
            .ip4 => {
                self.printStatus(.info, "Control connection is IPv4, trying PASV...", .{});
                const data_addr_ip4 = self.pasv(reader, writer, arena) catch |e| {
                    self.printStatus(.err, "{}: PASV failed", .{e});
                    return switch (e) {
                        error.ConnectionFailed => error.ConnectionFailed,
                        error.InvalidPasvResponse => error.CommandFailed,
                    };
                };
                const data_addr = std.Io.net.IpAddress{ .ip4 = data_addr_ip4 };
                break :ds data_addr.connect(self.io, .{ .mode = .stream }) catch |e| {
                    self.printStatus(.err, "{}: Failed to connect to PASV address", .{e});
                    return error.ConnectionFailed;
                };
            },
            .ip6 => {
                self.printStatus(.info, "Control connection is IPv6, trying EPSV...", .{});
                const data_port = self.epsv(control_addr, reader, writer, arena) catch |e| return e;
                var data_addr = control_addr;
                data_addr.ip6.port = data_port;
                break :ds data_addr.connect(self.io, .{ .mode = .stream }) catch |e| {
                    self.printStatus(.err, "{}: Failed to connect to PASV address", .{e});
                    return error.ConnectionFailed;
                };
            },
        };
        errdefer data_stream.close(self.io);
        self.printStatus(.info, "Connecting to data address... Done.", .{});
        try self.retr(writer, reader, file_path, arena);
        return data_stream;
    }

    /// Download a file via FTP with progress reporting
    /// Protocol specification references:
    /// - https://datatracker.ietf.org/doc/html/rfc959
    /// - https://datatracker.ietf.org/doc/html/rfc2428 (for EPSV)
    /// - https://datatracker.ietf.org/doc/html/rfc3659 (for SIZE)
    fn download(self: *Self) FtpError!u64 {
        // Arena for all FTP response message allocations
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        // Parse URL to get host and port
        const uri = std.Uri.parse(self.url) catch |e| {
            self.printStatus(.err, "{}: Invalid URL format", .{e});
            return error.InvalidUrl;
        };
        if (!std.mem.eql(u8, uri.scheme, "ftp")) {
            self.printStatus(.err, "Not an FTP URL", .{});
            return error.InvalidUrl;
        }
        var host_buf: [std.Io.net.HostName.max_len]u8 = undefined;
        const host = uri.getHost(&host_buf) catch |e| {
            self.printStatus(.err, "{}: Host name missing from URL", .{e});
            return error.InvalidUrl;
        };
        const port: u16 = uri.port orelse DEFAULT_PORT;

        // Connect and create stream reader & writer
        self.printStatus(.info, "Connecting...", .{});
        const control_stream = host.connect(self.io, port, .{ .mode = .stream }) catch |e| {
            self.printStatus(.err, "{}: Connection failed", .{e});
            return error.ConnectionFailed;
        };
        defer control_stream.close(self.io);
        self.printStatus(.info, "Connecting... Done. address: {f}", .{control_stream.socket.address});

        var read_buf: [4096]u8 = undefined;
        var stream_reader = control_stream.reader(self.io, &read_buf);
        const reader = &stream_reader.interface;
        var write_buf: [512]u8 = undefined;
        var stream_writer = control_stream.writer(self.io, &write_buf);
        const writer = &stream_writer.interface;

        try self.greeting(reader, allocator);

        self.printStatus(.info, "Logging in...", .{});
        try self.login(reader, writer, allocator);
        self.printStatus(.info, "Logging in... Done.", .{});

        const system_type = try self.syst(reader, writer, allocator);
        self.printStatus(.info, "Server's system type: {s}", .{system_type});

        const query = if (uri.query) |q| q.percent_encoded else null;
        try self.set_transfer_type(query, reader, writer, allocator);

        const file_path = uri.path.percent_encoded;
        const file_size_opt = self.get_file_size(file_path, reader, writer, allocator);
        if (file_size_opt) |file_size| {
            self.printStatus(.info, "File size: {d} bytes", .{file_size});
        } else {
            self.printStatus(.info, "File size: unknown", .{});
        }

        self.printStatus(.info, "Starting file transfer...", .{});
        const data_stream = try self.start_file_transfer(
            control_stream.socket.address,
            file_path,
            reader,
            writer,
            allocator,
        );
        errdefer data_stream.close(self.io);

        // Open output file
        const file = blk: {
            if (std.fs.path.isAbsolute(self.output_path)) {
                break :blk std.Io.Dir.createFileAbsolute(self.io, self.output_path, .{}) catch |e| {
                    self.printStatus(.err, "{}: Cannot create file", .{e});
                    return error.FileCreateFailed;
                };
            } else {
                break :blk std.Io.Dir.cwd().createFile(self.io, self.output_path, .{}) catch |e| {
                    self.printStatus(.err, "{}: Cannot create file", .{e});
                    return error.FileCreateFailed;
                };
            }
        };
        defer file.close(self.io);

        // Transfer
        const buffer_size = 64 * 1024; // Large buffer size for better performance
        var ds_buffer: [buffer_size]u8 = undefined;
        var ds_reader = data_stream.reader(self.io, &ds_buffer);
        const ds_reader_interface = &ds_reader.interface;

        var file_buffer: [buffer_size]u8 = undefined;
        var file_writer = file.writer(self.io, &file_buffer);
        const file_writer_interface = &file_writer.interface;

        self.printStatus(.info, "Start downloading...", .{});
        var downloaded: usize = 0;
        if (file_size_opt) |file_size| {
            // Chunk-based reading with progress updates. We require the server to send exactly file_size bytes, otherwise it's an error.
            const chunk_size = 8192;
            while (downloaded < file_size) {
                const to_read = @min(chunk_size, file_size - downloaded);
                ds_reader_interface.streamExact(file_writer_interface, to_read) catch |e| {
                    // error.EndOfStream happening here means that the actual number of bytes we get is less than the file size.
                    // We require exactly file_size bytes, so error.EndOfStream is an error case.
                    switch (e) {
                        error.EndOfStream => self.printStatus(.err, "{}: Incomplete transfer (server sent fewer bytes than expected)", .{e}),
                        error.ReadFailed => self.printStatus(.err, "{}: Network read error", .{e}),
                        error.WriteFailed => self.printStatus(.err, "{}: Disk write error", .{e}),
                    }
                    return error.TransferFailed;
                };
                downloaded +|= to_read;
                self.printStatus(.{ .progress = .{ .downloaded = downloaded, .total = file_size } }, "", .{});
            }
        } else {
            // If file size is unknown, just stream until EOF without progress updates
            self.printStatus(.info, "File size unknown, downloading without progress updates...", .{});
            while (true) {
                const transferred = ds_reader_interface.stream(file_writer_interface, .unlimited) catch |e| {
                    self.printStatus(.err, "{}: Transfer error", .{e});
                    return error.TransferFailed;
                };
                if (transferred == 0) break; // End of stream
                downloaded +|= transferred;
            }
        }
        // Don't forget to flush!
        file_writer_interface.flush() catch |e| self.printStatus(.err, "{}: Disk write error", .{e});

        // 226 complete
        // Close data stream first before reading the final response,
        // as some servers may wait for the client to close the data connection before sending the completion response.
        data_stream.close(self.io);
        _ = readResponse(reader, allocator) catch |e| {
            self.printStatus(.err, "{}: Failed to read transfer completion response", .{e});
            return error.ConnectionFailed;
        };

        self.printStatus(.{ .done = downloaded }, "", .{});

        return downloaded;
    }

    fn sendCommand(writer: *std.Io.Writer, cmd: []const u8, payload: []const []const u8) std.Io.Writer.Error!void {
        try writer.writeAll(cmd);
        for (payload) |part| {
            try writer.writeAll(" ");
            try writer.writeAll(part);
        }
        try writer.writeAll("\r\n");
        try writer.flush();
    }

    fn readResponse(reader: *std.Io.Reader, allocator: std.mem.Allocator) (std.Io.Reader.StreamError || std.mem.Allocator.Error)!Response {
        while (true) {
            // Accumulate into a growable allocation
            var line_writer = std.Io.Writer.Allocating.init(allocator);
            defer line_writer.deinit();

            // Stream up to (but not including) '\n' into line_writer
            _ = try reader.streamDelimiter(&line_writer.writer, '\n');
            // Toss the '\n' itself
            reader.toss(1);

            // Strip trailing \r
            const line = std.mem.trimEnd(u8, line_writer.written(), "\r");
            if (line.len < 4) continue;

            // If we can't parse the code on this line, just try for the next line.
            const code = std.fmt.parseInt(u16, line[0..3], 10) catch continue;

            if (line[3] == ' ') {
                const msg = try allocator.dupe(u8, line[4..]);
                return Response{ .code = code, .message = msg };
            }
            // NNN- continuation, loop and deinit this line's allocation
        }
    }

    /// Print status on this thread's designated terminal line.
    /// Just return immediately on any error, since this is just a best-effort status update and shouldn't interfere with the main download logic.
    fn printStatus(self: *const Self, status: Status, comptime fmt: []const u8, args: anytype) void {
        // Calculate how many lines to move up from current position
        const lines_up = self.task_count - self.task_id;

        // Lock stderr with a larger buffer for the entire operation
        var buffer: [512]u8 = undefined;
        var stderr = std.debug.lockStderr(&buffer);
        defer std.debug.unlockStderr();
        const writer = &stderr.file_writer.interface;

        // Move cursor up and clear line
        // \x1b[nA = move up n lines
        // \x1b[2K = clear entire line
        writer.print("\x1b[{d}A\x1b[2K\r{s: <25} ", .{ lines_up, self.filename }) catch return;

        switch (status) {
            .info => {
                writer.print(fmt, args) catch return;
            },
            .err => {
                writer.print("error: ", .{}) catch return;
                writer.print(fmt, args) catch return;
            },
            .progress => |p| {
                const percent: f64 = if (p.total > 0)
                    @as(f64, @floatFromInt(p.downloaded)) / @as(f64, @floatFromInt(p.total)) * 100
                else
                    0;

                // Build progress bar string
                const bar_width: usize = 25;
                const filled: usize = @intFromFloat(percent / 100 * @as(f64, @floatFromInt(bar_width)));

                var bar: [bar_width]u8 = undefined;
                for (0..bar_width) |i| {
                    if (i < filled) {
                        bar[i] = '=';
                    } else if (i == filled) {
                        bar[i] = '>';
                    } else {
                        bar[i] = ' ';
                    }
                }

                const downloaded_mb = @as(f64, @floatFromInt(p.downloaded)) / (1024 * 1024);
                const total_mb = @as(f64, @floatFromInt(p.total)) / (1024 * 1024);
                writer.print("[{s}] {d:>5.1}% ({d:.1}/{d:.1} MB)", .{ &bar, percent, downloaded_mb, total_mb }) catch return;
            },
            .done => |bytes| {
                const mb = @as(f64, @floatFromInt(bytes)) / (1024 * 1024);
                writer.print("[=========================] Done! {d} bytes ({d:.1} MB)", .{ bytes, mb }) catch return;
            },
        }

        // Move cursor back down
        // \x1b[nB = move down n lines
        writer.print("\x1b[{d}B\r", .{lines_up}) catch return;
    }
};
