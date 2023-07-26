workspace(name = "tinfer")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:tinfer.bzl", "tinfer_workspace")

tinfer_workspace()

# load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

# rules_proto_dependencies()

# load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

# # Don't use preinstalled tools to ensure builds are as hermetic as possible
# rules_foreign_cc_dependencies()

# http_archive(
#     name = "rules_cuda",
#     sha256 = "f80438bee9906e9ecb1a8a4ae2365374ac1e8a283897281a2db2fb7fcf746333",
#     strip_prefix = "runtime-b1c7cce21ba4661c17ac72421c6a0e2015e7bef3/third_party/rules_cuda",
#     urls = ["https://github.com/tensorflow/runtime/archive/b1c7cce21ba4661c17ac72421c6a0e2015e7bef3.tar.gz"],
# )

# load("@rules_cuda//cuda:dependencies.bzl", "rules_cuda_dependencies")

# rules_cuda_dependencies()

http_archive(
    name = "rules_cuda",
    sha256 = "dc1f4f704ca56e3d5edd973f98a45f0487d0f28c689d0a57ba236112148b1833",
    strip_prefix = "rules_cuda-v0.1.2",
    urls = ["https://github.com/bazel-contrib/rules_cuda/releases/download/v0.1.2/rules_cuda-v0.1.2.tar.gz"],
)
load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")
rules_cuda_dependencies()
register_detected_cuda_toolchains()

# load("@rules_cc//cc:repositories.bzl", "rules_cc_toolchains")

# rules_cc_toolchains()

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",

    # Replace the commit hash in both places (below) with the latest, rather than using the stale one here.
    # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/3dddf205a1f5cde20faf2444c1757abe0564ff4c.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-3dddf205a1f5cde20faf2444c1757abe0564ff4c",
    # When you first run this tool, it'll recommend a sha256 hash to put here with a message like: "DEBUG: Rule 'hedron_compile_commands' indicated that a canonical reproducible form can be obtained by modifying arguments sha256 = ..."
)
load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()
