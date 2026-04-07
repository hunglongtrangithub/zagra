//! Simple FTP client for downloading files
//! Supports anonymous login and passive mode transfers

const std = @import("std");
const builtin = @import("builtin");

pub const FtpError = error{
    InvalidUrl,
    ConnectionFailed,
    LoginFailed,
    CommandFailed,
    InvalidResponse,
    FileNotFound,
    TransferFailed,
    SizeMismatch,
    FileCreateFailed,
    ThreadSpawnFailed,
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

/// Download multiple files in parallel with progress display
/// Returns slice of results for each file, in the same order as input items
/// Caller must free the returned slice with allocator.free()
/// Returns null on allocation failure
pub fn downloadFiles(allocator: std.mem.Allocator, items: []const DownloadItem) ?[]DownloadResult {
    if (items.len == 0) return &.{};

    const total_threads = std.math.cast(u8, items.len) orelse {
        std.debug.print("Too many files to download (max 255)\n", .{});
        return null;
    };

    // Allocate results and contexts
    const results = allocator.alloc(DownloadResult, items.len) catch {
        std.debug.print("Out of memory\n", .{});
        return null;
    };
    const contexts = allocator.alloc(DownloadContext, items.len) catch {
        std.debug.print("Out of memory\n", .{});
        allocator.free(results);
        return null;
    };
    defer allocator.free(contexts);
    const threads = allocator.alloc(std.Thread, items.len) catch {
        std.debug.print("Out of memory\n", .{});
        allocator.free(results);
        return null;
    };
    defer allocator.free(threads);

    // Initialize contexts
    for (items, 0..) |item, i| {
        results[i] = error.ConnectionFailed; // Default to error, will be overwritten
        contexts[i] = .{
            .allocator = allocator,
            .url = item.url,
            .output_path = item.output_path,
            .filename = getFilename(item.url),
            // items have <= 255 entries, so this cast is safe
            .thread_id = @intCast(i),
            .total_threads = total_threads,
            .result = &results[i],
        };
    }

    // Reserve lines for progress bars
    for (0..total_threads) |_| std.debug.print("\n", .{});

    enableWindowsAnsi();
    // Hide cursor during downloads
    std.debug.print("\x1b[?25l", .{});
    defer std.debug.print("\x1b[?25h\n", .{}); // Show cursor when done

    for (contexts, 0..) |*ctx, i| {
        threads[i] = std.Thread.spawn(.{}, DownloadContext.run, .{ctx}) catch {
            ctx.printStatus(.{ .err = "Failed to spawn thread" });
            ctx.result.* = error.ThreadSpawnFailed;
            // Wait for already-spawned threads
            for (threads[0..i]) |*t| {
                t.join();
            }
            return results;
        };
    }
    // Wait for all successfully spawned tasks
    for (threads) |*t| {
        t.join();
    }

    return results;
}

/// Enable ANSI escape code processing on Windows 10+ to allow cursor movement and line clearing
/// Note: haven't tested this on Winddows yet
fn enableWindowsAnsi() void {
    if (builtin.os.tag != .windows) return;
    const handle = std.os.windows.GetStdHandle(std.os.windows.STD_ERROR_HANDLE) catch return;
    var mode: std.os.windows.DWORD = 0;
    if (std.os.windows.kernel32.GetConsoleMode(handle, &mode) == 0) return;
    _ = std.os.windows.kernel32.SetConsoleMode(handle, mode | 0x0004); // ENABLE_VIRTUAL_TERMINAL_PROCESSING
}

fn getFilename(url: []const u8) []const u8 {
    // Find last '/' and return everything after it
    if (std.mem.lastIndexOf(u8, url, "/")) |idx| {
        return url[idx + 1 ..];
    }
    return url;
}

/// Download context containing all state for a single download thread
const DownloadContext = struct {
    allocator: std.mem.Allocator,
    url: []const u8,
    output_path: []const u8,
    filename: []const u8,
    thread_id: u8,
    total_threads: u8,
    result: *DownloadResult,

    const DEFAULT_PORT: u16 = 21;

    const Response = struct {
        code: u16,
        /// owned by caller
        message: []const u8,
    };

    const Self = @This();

    /// Thread entry point
    fn run(self: *Self) void {
        enableWindowsAnsi();
        self.result.* = self.download();
    }

    /// Download a file via FTP with progress reporting
    fn download(self: *Self) FtpError!u64 {
        // Arena for all FTP response message allocations
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        // 1. Parse URL
        const uri = std.Uri.parse(self.url) catch {
            self.printStatus(.{ .err = "Invalid URL" });
            return error.InvalidUrl;
        };

        if (!std.mem.eql(u8, uri.scheme, "ftp")) {
            self.printStatus(.{ .err = "Not an FTP URL" });
            return error.InvalidUrl;
        }

        self.printStatus(.{ .msg = "Connecting..." });

        var host_buf: [std.Uri.host_name_max]u8 = undefined;
        const host = uri.getHost(&host_buf) catch {
            self.printStatus(.{ .err = "Invalid host" });
            return error.InvalidUrl;
        };

        const port: u16 = uri.port orelse DEFAULT_PORT;

        // 2. Connect
        const addr_list = std.net.getAddressList(allocator, host, port) catch {
            self.printStatus(.{ .err = "DNS resolution failed" });
            return error.ConnectionFailed;
        };
        if (addr_list.addrs.len == 0) {
            self.printStatus(.{ .err = "Unknown host" });
            return error.ConnectionFailed;
        }
        const server_addr = addr_list.addrs[0];

        var control = std.net.tcpConnectToAddress(server_addr) catch {
            self.printStatus(.{ .err = "Connection failed" });
            return error.ConnectionFailed;
        };
        defer control.close();

        // Create persistent reader and writer
        var read_buf: [4096]u8 = undefined;
        var stream_reader = control.reader(&read_buf);
        const reader: *std.Io.Reader = stream_reader.interface();

        var write_buf: [512]u8 = undefined;
        var stream_writer = control.writer(&write_buf);
        const writer: *std.Io.Writer = &stream_writer.interface;

        // 3. Welcome
        const welcome = readResponse(reader, allocator) catch {
            self.printStatus(.{ .err = "No welcome message" });
            return error.ConnectionFailed;
        };
        if (welcome.code != 220) {
            self.printStatus(.{ .err = "Server rejected connection" });
            return error.ConnectionFailed;
        }

        self.printStatus(.{ .msg = "Logging in..." });

        // 4. USER
        sendCommand(writer, &.{"USER anonymous"}) catch {
            self.printStatus(.{ .err = "Failed to send USER" });
            return error.ConnectionFailed;
        };
        const user_resp = readResponse(reader, allocator) catch {
            self.printStatus(.{ .err = "No USER response" });
            return error.ConnectionFailed;
        };
        if (user_resp.code != 331 and user_resp.code != 230) {
            self.printStatus(.{ .err = "USER rejected" });
            return error.LoginFailed;
        }

        // 5. PASS
        if (user_resp.code == 331) {
            sendCommand(writer, &.{"PASS anonymous@example.com"}) catch {
                self.printStatus(.{ .err = "Failed to send PASS" });
                return error.ConnectionFailed;
            };
            const pass_resp = readResponse(reader, allocator) catch {
                self.printStatus(.{ .err = "No PASS response" });
                return error.ConnectionFailed;
            };
            if (pass_resp.code != 230) {
                self.printStatus(.{ .err = "Login failed" });
                return error.LoginFailed;
            }
        }

        // 6. TYPE I
        sendCommand(writer, &.{"TYPE I"}) catch {
            self.printStatus(.{ .err = "Failed to send TYPE" });
            return error.ConnectionFailed;
        };
        const type_resp = readResponse(reader, allocator) catch {
            self.printStatus(.{ .err = "No TYPE response" });
            return error.ConnectionFailed;
        };
        if (type_resp.code != 200) {
            self.printStatus(.{ .err = "TYPE I failed" });
            return error.CommandFailed;
        }

        self.printStatus(.{ .msg = "Getting file size..." });

        const file_path = uri.path.percent_encoded;

        // 7. SIZE
        sendCommand(writer, &.{ "SIZE ", file_path }) catch {
            self.printStatus(.{ .err = "Failed to send SIZE" });
            return error.ConnectionFailed;
        };
        self.printStatus(.{ .msg = "SIZE sent, waiting for response..." });
        const size_resp = readResponse(reader, allocator) catch {
            self.printStatus(.{ .err = "No SIZE response" });
            return error.ConnectionFailed;
        };
        if (size_resp.code != 213) {
            self.printStatus(.{ .err = "File not found" });
            return error.FileNotFound;
        }
        const file_size = std.fmt.parseInt(u64, size_resp.message, 10) catch {
            self.printStatus(.{ .err = "Invalid file size" });
            return error.InvalidResponse;
        };

        // 8. EPSV / PASV
        sendCommand(writer, &.{"EPSV"}) catch {
            self.printStatus(.{ .err = "Failed to send EPSV" });
            return error.ConnectionFailed;
        };
        self.printStatus(.{ .msg = "EPSV sent, waiting for response..." });
        const epsv_resp = readResponse(reader, allocator) catch {
            self.printStatus(.{ .err = "No EPSV response" });
            return error.ConnectionFailed;
        };

        const data_addr = blk: {
            if (epsv_resp.code == 229) {
                if (parseEpsvResponse(epsv_resp.message, server_addr)) |addr| {
                    break :blk addr;
                } else |_| {
                    self.printStatus(.{ .msg = "EPSV parse failed, falling back to PASV..." });
                    sendCommand(writer, &.{"PASV"}) catch {
                        self.printStatus(.{ .err = "Failed to send PASV" });
                        return error.ConnectionFailed;
                    };
                    const pasv_resp = readResponse(reader, allocator) catch {
                        self.printStatus(.{ .err = "No PASV response" });
                        return error.ConnectionFailed;
                    };
                    if (pasv_resp.code != 227) {
                        self.printStatus(.{ .err = "PASV failed" });
                        return error.CommandFailed;
                    }
                    break :blk parsePasvResponse(pasv_resp.message) catch {
                        self.printStatus(.{ .err = "Invalid PASV response" });
                        return error.InvalidResponse;
                    };
                }
            } else if (epsv_resp.code == 500) {
                sendCommand(writer, &.{"PASV"}) catch {
                    self.printStatus(.{ .err = "Failed to send PASV" });
                    return error.ConnectionFailed;
                };
                const pasv_resp = readResponse(reader, allocator) catch {
                    self.printStatus(.{ .err = "No PASV response" });
                    return error.ConnectionFailed;
                };
                if (pasv_resp.code != 227) {
                    self.printStatus(.{ .err = "PASV failed" });
                    return error.CommandFailed;
                }
                break :blk parsePasvResponse(pasv_resp.message) catch {
                    self.printStatus(.{ .err = "Invalid PASV response" });
                    return error.InvalidResponse;
                };
            } else {
                self.printStatus(.{ .err = "EPSV failed" });
                return error.CommandFailed;
            }
        };

        // 9. Connect to data port
        var data_stream = std.net.tcpConnectToAddress(data_addr) catch {
            self.printStatus(.{ .err = "Data connection failed" });
            return error.ConnectionFailed;
        };
        defer data_stream.close();

        // 10. RETR
        sendCommand(writer, &.{ "RETR ", file_path }) catch {
            self.printStatus(.{ .err = "Failed to send RETR" });
            return error.ConnectionFailed;
        };
        const retr_resp = readResponse(reader, allocator) catch {
            self.printStatus(.{ .err = "No RETR response" });
            return error.ConnectionFailed;
        };
        if (retr_resp.code != 150 and retr_resp.code != 125) {
            self.printStatus(.{ .err = "RETR failed" });
            return error.TransferFailed;
        }

        // 11. Open output file
        const file = blk: {
            if (std.fs.path.isAbsolute(self.output_path)) {
                break :blk std.fs.createFileAbsolute(self.output_path, .{}) catch {
                    self.printStatus(.{ .err = "Cannot create file" });
                    return error.FileCreateFailed;
                };
            } else {
                break :blk std.fs.cwd().createFile(self.output_path, .{}) catch {
                    self.printStatus(.{ .err = "Cannot create file" });
                    return error.FileCreateFailed;
                };
            }
        };
        defer file.close();

        // 12. Transfer
        const buffer_size = 64 * 1024; // Large buffer size for better performance
        var ds_buffer: [buffer_size]u8 = undefined;
        var ds_reader = data_stream.reader(&ds_buffer);
        const ds_reader_interface: *std.Io.Reader = ds_reader.interface();

        var file_buffer: [buffer_size]u8 = undefined;
        var file_writer = file.writer(&file_buffer);
        const file_writer_interface = &file_writer.interface;

        self.printStatus(.{ .msg = "Start downloading..." });
        var downloaded: usize = 0;
        const chunk_size = 8192; // Small chunk size for frequent UI update
        while (downloaded < file_size) {
            const to_read = @min(chunk_size, file_size - downloaded);
            ds_reader_interface.streamExact(file_writer_interface, to_read) catch |e| {
                // error.EndOfStream happening here means that the actual number of bytes we get is less than the file size.
                // We require exactly file_size bytes, so error.EndOfStream is an error case.
                switch (e) {
                    error.EndOfStream => self.printStatus(.{ .err = "Incomplete transfer (server sent fewer bytes than expected)" }),
                    error.ReadFailed => self.printStatus(.{ .err = "Network read error" }),
                    error.WriteFailed => self.printStatus(.{ .err = "Disk write error" }),
                }
                return error.TransferFailed;
            };
            downloaded += to_read;
            self.printStatus(.{ .progress = .{ .downloaded = downloaded, .total = file_size } });
        }
        // Don't forget to flush!
        file_writer_interface.flush() catch self.printStatus(.{ .err = "Disk write error" });

        // 13. 226 complete
        const complete_resp = readResponse(reader, allocator) catch {
            self.printStatus(.{ .err = "No completion response" });
            return error.ConnectionFailed;
        };
        _ = complete_resp;

        self.printStatus(.{ .done = downloaded });

        // 14. QUIT (best effort)
        sendCommand(writer, &.{"QUIT"}) catch {};
        _ = readResponse(reader, allocator) catch {};

        return downloaded;
    }

    fn sendCommand(writer: *std.Io.Writer, cmd: []const []const u8) error{ConnectionFailed}!void {
        for (cmd) |part| {
            writer.writeAll(part) catch return error.ConnectionFailed;
        }
        writer.writeAll("\r\n") catch return error.ConnectionFailed;
        writer.flush() catch return error.ConnectionFailed;
    }

    fn readResponse(reader: *std.Io.Reader, allocator: std.mem.Allocator) !Response {
        while (true) {
            // Accumulate into a growable allocation
            var line_writer = std.Io.Writer.Allocating.init(allocator);
            defer line_writer.deinit();

            // Stream up to (but not including) '\n' into line_writer
            _ = reader.streamDelimiter(&line_writer.writer, '\n') catch return error.ConnectionFailed;
            // Toss the '\n' itself
            reader.toss(1);

            // Strip trailing \r
            const line = std.mem.trimRight(u8, line_writer.written(), "\r");

            if (line.len < 4) continue;

            const code = std.fmt.parseInt(u16, line[0..3], 10) catch continue;

            if (line[3] == ' ') {
                const msg = try allocator.dupe(u8, line[4..]);
                return Response{ .code = code, .message = msg };
            }
            // NNN- continuation, loop and deinit this line's allocation
        }
    }

    /// Parse PASV response: "Entering Passive Mode (h1,h2,h3,h4,p1,p2)"
    fn parsePasvResponse(msg: []const u8) !std.net.Address {
        const start = std.mem.indexOf(u8, msg, "(") orelse return error.InvalidResponse;
        const end = std.mem.indexOf(u8, msg, ")") orelse return error.InvalidResponse;
        if (start >= end) return error.InvalidResponse;

        const nums_str = msg[start + 1 .. end];

        // Parse 6 comma-separated numbers
        var nums: [6]u8 = undefined;
        var iter = std.mem.splitScalar(u8, nums_str, ',');
        var i: usize = 0;
        while (iter.next()) |part| {
            if (i >= 6) return error.InvalidResponse;
            nums[i] = std.fmt.parseInt(u8, std.mem.trim(u8, part, " "), 10) catch return error.InvalidResponse;
            i += 1;
        }
        if (i != 6) return error.InvalidResponse;

        // Port = p1 * 256 + p2
        const port: u16 = @as(u16, nums[4]) * 256 + nums[5];

        return std.net.Address{ .in = std.net.Ip4Address.init(nums[0..4].*, port) };
    }

    fn parseEpsvResponse(msg: []const u8, server_addr: std.net.Address) !std.net.Address {
        const start = std.mem.indexOf(u8, msg, "|||") orelse return error.InvalidResponse;
        const end = std.mem.indexOfScalar(u8, msg[start + 3 ..], '|') orelse return error.InvalidResponse;
        const port_str = msg[start + 3 .. start + 3 + end];
        const port = std.fmt.parseInt(u16, port_str, 10) catch return error.InvalidResponse;

        var addr = server_addr;
        addr.setPort(port);
        return addr;
    }

    test parsePasvResponse {
        const addr = try parsePasvResponse("Entering Passive Mode (131,254,14,19,196,108)");
        const expected = std.net.Address{ .in = std.net.Ip4Address.init(.{ 131, 254, 14, 19 }, 50284) };
        try std.testing.expectEqual(expected, addr);
    }

    /// Print status on this thread's designated terminal line
    fn printStatus(self: *const DownloadContext, status: Status) void {
        // Calculate how many lines to move up from current position
        const lines_up = self.total_threads - self.thread_id;

        // Lock stderr with a larger buffer for the entire operation
        var buffer: [512]u8 = undefined;
        const writer = std.debug.lockStderrWriter(&buffer);
        defer std.debug.unlockStderrWriter();

        // Move cursor up and clear line
        // \x1b[nA = move up n lines
        // \x1b[2K = clear entire line
        writer.print("\x1b[{d}A\x1b[2K\r{s: <25} ", .{ lines_up, self.filename }) catch return;

        switch (status) {
            .msg => |msg| {
                writer.print("{s}", .{std.mem.trim(u8, msg, " \r\t\n")}) catch return;
            },
            .err => |err| {
                writer.print("ERROR: {s}", .{std.mem.trim(u8, err, " \r\t\n")}) catch return;
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

const Status = union(enum) {
    msg: []const u8,
    err: []const u8,
    progress: struct { downloaded: u64, total: u64 },
    done: u64,
};

/// For Testing
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create test directory with absolute path
    const test_dir = "/tmp/ftp_test";
    std.fs.makeDirAbsolute(test_dir) catch |e| switch (e) {
        error.PathAlreadyExists => {},
        else => {
            std.debug.print("Failed to create test directory: {}\n", .{e});
            return;
        },
    };

    const download_items = [_]DownloadItem{
        .{
            .url = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
            .output_path = "/tmp/ftp_test/siftsmall.tar.gz",
        },
    };

    std.debug.print("Testing FTP download to absolute path: {s}\n", .{test_dir});
    const results = downloadFiles(allocator, &download_items) orelse {
        std.debug.print("\nAllocation failed!\n", .{});
        return;
    };
    defer allocator.free(results);

    var all_success = true;
    for (results) |result| {
        if (result) |bytes| {
            std.debug.print("\nDownloaded {d} bytes\n", .{bytes});
        } else |err| {
            std.debug.print("\nDownload failed: {}\n", .{err});
            all_success = false;
        }
    }

    if (all_success) {
        std.debug.print("All downloads successful!\n", .{});
    }
}

test "getFilename" {
    try std.testing.expectEqualStrings("file.tar.gz", getFilename("ftp://example.com/path/to/file.tar.gz"));
    try std.testing.expectEqualStrings("file.txt", getFilename("/just/a/path/file.txt"));
    try std.testing.expectEqualStrings("nopath", getFilename("nopath"));
}
