load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")
load("@fbsource//xplat/executorch/backends/qualcomm/qnn_version.bzl", "get_qnn_library_version")

def define_common_targets():
    runtime.cxx_library(
        name = "runner_lib",
        srcs = glob(
            [
                "runner/*.cpp",
            ],
        ),
        exported_headers = glob([
            "runner/*.h",
        ]),
        compiler_flags = [
            "-Wno-global-constructors",
            "-Wunused-command-line-argument",
        ],
        deps = [
            "//executorch/extension/llm/runner:stats",
            "//executorch/extension/tensor:tensor",
             "//executorch/kernels/quantized:generated_lib",
            "fbsource//third-party/qualcomm/qnn/qnn-{0}:api".format(get_qnn_library_version()),
        ],
        exported_deps = [
            "//executorch/extension/module:module",
            "//executorch/extension/llm/sampler:sampler",
            "//executorch/examples/models/llama/tokenizer:tiktoken",
            "//executorch/extension/evalue_util:print_evalue",
            "//executorch/backends/qualcomm/runtime:runtime",
            "//executorch/extension/llm/runner:runner_lib",
            "//pytorch/tokenizers:llama2c_tokenizer",
            "//pytorch/tokenizers:hf_tokenizer",
            "//pytorch/tokenizers:regex_lookahead",
            "//pytorch/tokenizers:tiktoken",
        ],
        external_deps = [
            "gflags",
        ],
        platforms = [ANDROID],
        **get_oss_build_kwargs()
    )

    runtime.cxx_binary(
        name = "qnn_llama_runner",
        srcs = [
            "qnn_llama_runner.cpp",
        ],
        compiler_flags = [
            "-Wno-global-constructors",
        ],
        deps = [
            ":runner_lib",
            "//executorch/extension/threadpool:threadpool", # this depeneency shouldn't be needed. But it fails to build..
        ],
        external_deps = [
            "gflags",
        ],
        platforms = [ANDROID],
        **get_oss_build_kwargs()
    )
