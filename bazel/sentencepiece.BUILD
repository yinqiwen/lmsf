package(default_visibility = ["//visibility:public"])

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "sentencepiece",
    generate_args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DSNAPPY_BUILD_TESTS=OFF",
        "-DSNAPPY_BUILD_BENCHMARKS=OFF",
    ],
    lib_source = ":all_srcs",
    out_lib_dir = "lib64",
    out_static_libs = [
        "libsentencepiece.a",
    ],
    visibility = ["//visibility:public"],
)
