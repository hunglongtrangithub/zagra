const std = @import("std");
const config = @import("config.zig");

/// List of benchmarks in the bench directory
const benchmarks = [_][]const u8{
    "bench_vector_simd",
    "bench_nn_descent",
    "bench_optimizer",
};
comptime {
    for (benchmarks) |bench_name| {
        std.debug.assert(std.mem.startsWith(u8, bench_name, "bench_"));
    }
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create znpy dependency
    const znpy_dep = b.dependency("znpy", .{
        .target = target,
        .optimize = optimize,
    });
    const znpy_mod = znpy_dep.module("znpy");

    // Create zagra module
    const zagra_mod = b.addModule("zagra", .{
        .root_source_file = b.path(config.SRC_DIR ++ "/root.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "znpy", .module = znpy_mod },
        },
    });

    // Create zagra executable
    // Will be automatically installed with `zig build`
    const zagra_exe = b.addExecutable(.{
        .name = "zagra",
        .root_module = b.createModule(.{
            .root_source_file = b.path(config.SRC_DIR ++ "/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zagra", .module = zagra_mod },
                .{ .name = "znpy", .module = znpy_mod },
            },
        }),
    });
    b.installArtifact(zagra_exe);

    // Create run step for the main executable
    // Binary is installed first before running
    const zagra_run_cmd = b.addRunArtifact(zagra_exe);
    if (b.args) |args| zagra_run_cmd.addArgs(args);
    zagra_run_cmd.step.dependOn(b.getInstallStep());

    const zagra_run_step = b.step("run", "Run the app");
    zagra_run_step.dependOn(&zagra_run_cmd.step);

    // Add test steps for both the zagra module and the executable
    const zagra_mod_tests = b.addTest(.{
        .root_module = zagra_mod,
    });
    const run_zagra_mod_tests = b.addRunArtifact(zagra_mod_tests);
    const zagra_exe_tests = b.addTest(.{
        .root_module = zagra_exe.root_module,
    });
    const run_zagra_exe_tests = b.addRunArtifact(zagra_exe_tests);

    // Create config module
    // Shared configuration for benchmarks
    const config_mod = b.addModule("config", .{
        .root_source_file = b.path("config.zig"),
        .target = target,
    });

    //  Create bench module for testing
    const bench_mod = b.addModule("bench", .{
        .root_source_file = b.path(config.BENCH_DIR ++ "/root.zig"),
        .target = target,
    });
    const bench_mod_tests = b.addTest(.{
        .root_module = bench_mod,
    });
    const run_bench_mod_tests = b.addRunArtifact(bench_mod_tests);

    // Create benchmark executables and run steps
    // Binary is installed first before running
    inline for (benchmarks) |bench_name| {
        const bench_exe = b.addExecutable(.{
            .name = bench_name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(config.BENCH_DIR ++ "/" ++ bench_name ++ ".zig"),
                .target = target,
                // Always compile to ReleaseFast
                .optimize = std.builtin.OptimizeMode.ReleaseFast,
                .imports = &.{
                    .{ .name = "zagra", .module = zagra_mod },
                    .{ .name = "config", .module = config_mod },
                },
            }),
        });
        const install_step = b.addInstallArtifact(bench_exe, .{
            .dest_dir = .default,
        });

        const bench_cmd = b.addRunArtifact(bench_exe);
        if (b.args) |args| bench_cmd.addArgs(args);

        const bench_step = b.step(bench_name, "Run the " ++ bench_name ++ " bench");
        bench_step.dependOn(&bench_cmd.step);
        bench_step.dependOn(&install_step.step);
    }

    // Add a step to list all benchmarks
    const list_benchmarks_step = b.step("bench", "List all benchmarks");
    list_benchmarks_step.makeFn = struct {
        fn make(_: *std.Build.Step, _: std.Build.Step.MakeOptions) anyerror!void {
            var stdout_buffer: [1024]u8 = undefined;
            var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
            const stdout = &stdout_writer.interface;

            try stdout.print("Available benchmarks:\n", .{});
            inline for (benchmarks) |bench_name| {
                try stdout.print("- {s}\n", .{bench_name});
            }
            try stdout.print("Use `zig build <benchmark name> -- [args]` to run the benchmark.\n", .{});
            try stdout.flush();
        }
    }.make;

    // Add executable for dataset downloader
    const texmex_exe = b.addExecutable(.{
        .name = "textmexsteal",
        .root_module = b.createModule(.{
            .root_source_file = b.path(config.DATA_DIR ++ "/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "config", .module = config_mod },
                .{ .name = "znpy", .module = znpy_mod },
            },
        }),
    });
    const texmex_run_cmd = b.addRunArtifact(texmex_exe);
    if (b.args) |args| texmex_run_cmd.addArgs(args);
    const texmex_run_step = b.step("texmex", "Run the TEXMEX ANN vector set downloader");
    texmex_run_step.dependOn(&texmex_run_cmd.step);
    const texmex_exe_tests = b.addTest(.{
        .root_module = texmex_exe.root_module,
    });
    const run_texmex_exe_tests = b.addRunArtifact(texmex_exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_zagra_mod_tests.step);
    test_step.dependOn(&run_zagra_exe_tests.step);
    test_step.dependOn(&run_bench_mod_tests.step);
    test_step.dependOn(&run_texmex_exe_tests.step);
}
