load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def tinfer_workspace(path_prefix = "", tf_repo_name = "", **kwargs):
    http_archive(
        name = "bazel_skylib",
        urls = [
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
        ],
        sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
    )
    rules_foreign_cc_ver = kwargs.get("rules_foreign_cc_ver", "0.8.0")
    http_archive(
        name = "rules_foreign_cc",
        strip_prefix = "rules_foreign_cc-{ver}".format(ver = rules_foreign_cc_ver),
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/{ver}.zip".format(ver = rules_foreign_cc_ver),
    )

    # rules_proto defines abstract rules for building Protocol Buffers.
    http_archive(
        name = "rules_proto",
        sha256 = "2490dca4f249b8a9a3ab07bd1ba6eca085aaf8e45a734af92aad0c42d9dc7aaf",
        strip_prefix = "rules_proto-218ffa7dfa5408492dc86c01ee637614f8695c45",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/218ffa7dfa5408492dc86c01ee637614f8695c45.tar.gz",
            "https://github.com/bazelbuild/rules_proto/archive/218ffa7dfa5408492dc86c01ee637614f8695c45.tar.gz",
        ],
    )

    gtest_ver = kwargs.get("gtest_ver", "1.11.0")
    gtest_name = "googletest-release-{ver}".format(ver = gtest_ver)
    http_archive(
        name = "com_google_googletest",
        strip_prefix = gtest_name,
        urls = [
            "https://github.com/google/googletest/archive/release-{ver}.tar.gz".format(ver = gtest_ver),
        ],
    )

    protobuf_ver = kwargs.get("protobuf_ver", "3.15.8")
    protobuf_name = "protobuf-{ver}".format(ver = protobuf_ver)
    http_archive(
        name = "com_google_protobuf",
        strip_prefix = protobuf_name,
        urls = [
            "https://mirrors.tencent.com/github.com/protocolbuffers/protobuf/archive/v{ver}.tar.gz".format(ver = protobuf_ver),
            "https://github.com/protocolbuffers/protobuf/archive/v{ver}.tar.gz".format(ver = protobuf_ver),
        ],
    )

    abseil_ver = kwargs.get("abseil_ver", "20230125.3")
    abseil_name = "abseil-cpp-{ver}".format(ver = abseil_ver)
    http_archive(
        name = "com_google_absl",
        strip_prefix = abseil_name,
        urls = [
            "https://mirrors.tencent.com/github.com/abseil/abseil-cpp/archive/{ver}.tar.gz".format(ver = abseil_ver),
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/{ver}.tar.gz".format(ver = abseil_ver),
        ],
    )

    re2_ver = kwargs.get("re2_ver", "2023-07-01")
    re2_name = "re2-{ver}".format(ver = re2_ver)
    http_archive(
        name = "com_google_re2",
        strip_prefix = re2_name,
        urls = [
            "https://mirrors.tencent.com/github.com/google/re2/archive/{ver}.tar.gz".format(ver = re2_ver),
            "https://github.com/abseil/google/re2/refs/tags/{ver}.tar.gz".format(ver = re2_ver),
        ],
    )

    sentencepiece_ver = kwargs.get("sentencepiece_ver", "0.1.99")
    sentencepiece_name = "sentencepiece-{ver}".format(ver = sentencepiece_ver)
    http_archive(
        name = "com_google_sentencepiece",
        strip_prefix = sentencepiece_name,
        build_file = clean_dep("//:bazel/sentencepiece.BUILD"),
        urls = [
            "https://mirrors.tencent.com/github.com/google/sentencepiece/archive/v{ver}.tar.gz".format(ver = sentencepiece_ver),
            "https://github.com/google/sentencepiece/archive/refs/tags/v{ver}.tar.gz".format(ver = sentencepiece_ver),
        ],
    )

    _CUTLASS_BUILD_FILE = """
cc_library(
    name = "cutlass",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
"""

    cutlass_ver = kwargs.get("cutlass_ver", "2.10.0")
    cutlass_name = "cutlass-{ver}".format(ver = cutlass_ver)
    http_archive(
        name = "com_nvidia_cutlass",
        strip_prefix = cutlass_name,
        build_file_content = _CUTLASS_BUILD_FILE,
        urls = [
            "https://mirrors.tencent.com/github.com/NVIDIA/cutlass/archive/v{ver}.tar.gz".format(ver = cutlass_ver),
            "https://github.com/NVIDIA/cutlass/archive/refs/tags/v{ver}.tar.gz".format(ver = cutlass_ver),
        ],
    )

    _SPDLOG_BUILD_FILE = """
cc_library(
    name = "spdlog",
    hdrs = glob([
        "include/**/*.h",
    ]),
    srcs= glob([
        "src/*.cpp",
    ]),
    defines = ["SPDLOG_FMT_EXTERNAL", "SPDLOG_COMPILED_LIB"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)
"""
    spdlog_ver = kwargs.get("spdlog_ver", "1.10.0")
    spdlog_name = "spdlog-{ver}".format(ver = spdlog_ver)
    http_archive(
        name = "com_github_spdlog",
        strip_prefix = spdlog_name,
        urls = [
            "https://mirrors.tencent.com/github.com/gabime/spdlog/archive/v{ver}.tar.gz".format(ver = spdlog_ver),
            "https://github.com/gabime/spdlog/archive/v{ver}.tar.gz".format(ver = spdlog_ver),
        ],
        build_file_content = _SPDLOG_BUILD_FILE,
    )

    toml11_ver = kwargs.get("toml11_ver", "3.6.0")
    toml11_name = "toml11-{ver}".format(ver = toml11_ver)
    http_archive(
        name = "com_github_toml11",
        strip_prefix = toml11_name,
        urls = [
            "https://mirrors.tencent.com/github.com/ToruNiina/toml11/archive/v{ver}.tar.gz".format(ver = toml11_ver),
            "https://github.com/ToruNiina/toml11/archive/v{ver}.tar.gz".format(ver = toml11_ver),
        ],
        build_file = clean_dep("//:bazel/toml11.BUILD"),
    )

    _RAPIDJSON_BUILD_FILE = """
cc_library(
    name = "rapidjson",
    hdrs = glob(["include/rapidjson/**/*.h"]),
    includes = ["include"],
    defines = ["RAPIDJSON_HAS_STDSTRING=1"],
    visibility = [ "//visibility:public" ],
)
"""
    rapidjson_ver = kwargs.get("rapidjson_ver", "1.1.0")
    rapidjson_name = "rapidjson-{ver}".format(ver = rapidjson_ver)
    http_archive(
        name = "com_github_tencent_rapidjson",
        strip_prefix = rapidjson_name,
        urls = [
            "https://mirrors.tencent.com/github.com/Tencent/rapidjson/archive/v{ver}.tar.gz".format(ver = rapidjson_ver),
            "https://github.com/Tencent/rapidjson/archive/v{ver}.tar.gz".format(ver = rapidjson_ver),
        ],
        build_file_content = _RAPIDJSON_BUILD_FILE,
    )

    git_repository(
        name = "kcfg",
        remote = "https://github.com/yinqiwen/kcfg.git",
        branch = "master",
    )

    http_archive(
        name = "icu",
        strip_prefix = "icu-release-64-2",
        sha256 = "dfc62618aa4bd3ca14a3df548cd65fe393155edd213e49c39f3a30ccd618fc27",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/unicode-org/icu/archive/release-64-2.zip",
            "https://github.com/unicode-org/icu/archive/release-64-2.zip",
        ],
        build_file = clean_dep("//third_party/icu:BUILD.bzl"),
        patches = [clean_dep("//third_party/icu:udata.patch")],
        patch_args = ["-p1"],
    )

    jemalloc_ver = kwargs.get("jemalloc_ver", "5.3.0")
    jemalloc_name = "jemalloc-{ver}".format(ver = jemalloc_ver)
    http_archive(
        name = "com_github_jemalloc",
        strip_prefix = jemalloc_name,
        urls = [
            "https://github.com/jemalloc/jemalloc/releases/download/{ver}/jemalloc-{ver}.tar.bz2".format(ver = jemalloc_ver),
        ],
        build_file = clean_dep("//:bazel/jemalloc.BUILD"),
    )
