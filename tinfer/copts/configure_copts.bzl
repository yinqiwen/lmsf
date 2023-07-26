TINFER_DEFAULT_COPTS = [
    "-D__STDC_FORMAT_MACROS",
    "-D__STDC_LIMIT_MACROS",
    "-D__STDC_CONSTANT_MACROS",
    "-Werror=return-type",
]

TINFER_DEFAULT_CUDA_COPTS = [
    "-std=c++17",
]

TINFER_DEFAULT_LINKOPTS = [
    "-L/usr/local/lib64",
    "-lfmt",
    "-L/usr/local/cuda/lib64/",
    "-lcublas",
    "-lcublasLt",
    "-lcurand",
    "-lcrypto",
    "-lssl",
    "-ldl",
    "-lrt",
]
