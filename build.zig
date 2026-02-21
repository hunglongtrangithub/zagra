const std = @import("std");

const benches = [_][]const u8{
    "vector_simd",
    "nn_descent",
    "detour_count",
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const znpy_dep = b.dependency("znpy", .{
        .target = target,
        .optimize = optimize,
    });

    const znpy_mod = znpy_dep.module("znpy");

    const zagra_mod = b.addModule("zagra", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "znpy", .module = znpy_mod },
        },
    });

    inline for (benches) |bench_name| {
        const bench_exe = b.addExecutable(.{
            .name = bench_name,
            .root_module = b.createModule(.{
                .root_source_file = b.path("benches/" ++ bench_name ++ ".zig"),
                .target = target,
                // Always compile to ReleaseFast
                .optimize = std.builtin.OptimizeMode.ReleaseFast,
                .imports = &.{
                    .{ .name = "zagra", .module = zagra_mod },
                },
            }),
        });

        const run_cmd = b.addRunArtifact(bench_exe);
        const run_step = b.step("bench_" ++ bench_name, "Run the " ++ bench_name ++ " bench");
        run_step.dependOn(&run_cmd.step);
    }

    const exe = b.addExecutable(.{
        .name = "zagra",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zagra", .module = zagra_mod },
                .{ .name = "znpy", .module = znpy_mod },
            },
        }),
    });

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = zagra_mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
