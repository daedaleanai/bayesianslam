COPTS = [
    "-I/usr/local/include",
    "-Iexternal/gtest/include",
    "-std=c++11",
]

LINKOPTS = [
    "-lm",
    "-lpng",
    "-lprofiler",
]

cc_library(
    name = "slam2d",
    srcs = [
        "aircraft.cc",
        "aircraft.hh",
        "object_sampler.cc",
        "object_sampler.hh",
        "objects.cc",
        "objects.hh",
        "region.cc",
        "region.hh",
        "screen2d.cc",
        "screen2d.hh",
        "simulation2d.cc",
        "simulation2d.hh",
    ],
    hdrs = [
        "aircraft.hh",
        "objects.hh",
        "point.hh",
        "screen2d.hh",
        "simulation2d.hh",
    ],
    copts = COPTS,
    deps = ["@boom"],
)

# To build in debug mode do bazel build --spawn_strategy=standalone -c dbg slam2d.
# genrule(
#     name = "slam2d_dsym",
#     srcs = [":slam2d"],
#     outs = ["slam2d.dSYM"],
#     cmd = "dsymutil $(locations :slam2d) -o $@",
#     output_to_bindir = True,
# )

cc_binary(
    name = "flight2d",
    srcs = ["flight2d_main.cc"],
    copts = COPTS,
    linkopts = LINKOPTS,
    deps = [
        ":slam2d",
        "@boom",
    ],
)

# A simulation to identify the fixed objects in a 2D scene, without movement.
cc_binary(
    name = "static2d",
    srcs = ["static2d_main.cc"],
    copts = COPTS,
    linkopts = LINKOPTS,
    deps = [
        ":slam2d",
        "@boom",
    ],
)

# A simulation to identify the fixed objects in a 2D scene, without movement.
cc_binary(
    name = "static2d-mcmc",
    srcs = ["static2d-mcmc_main.cc"],
    copts = COPTS,
    linkopts = LINKOPTS,
    deps = [
        ":slam2d",
        "@boom",
    ],
)

genrule(
    name = "static2d-mcmc_dsym",
    srcs = [":static2d-mcmc"],
    outs = ["static2d-mcmc.dSYM"],
    cmd = "dsymutil $(location :static2d-mcmc) -o $@",
    output_to_bindir = True,
)

cc_test(
    name = "box_union_test",
    srcs = ["box_union_test.cc"],
    copts = COPTS,
    linkopts = LINKOPTS,
    deps = [
        ":slam2d",
        "@boom",
        "@gtest",
    ],
)

cc_test(
    name = "mcmc_test",
    srcs = ["mcmc_test.cc"],
    copts = COPTS,
    linkopts = LINKOPTS,
    deps = [
        ":slam2d",
        "@boom",
        "@gtest",
    ],
)

genrule(
    name = "mcmc_test_dsym",
    testonly = True,
    srcs = [":mcmc_test"],
    outs = ["mcmc_test.dSYM"],
    cmd = "dsymutil $(location :mcmc_test) -o $@",
    output_to_bindir = True,
)

cc_test(
    name = "objects_test",
    srcs = ["objects_test.cc"],
    copts = COPTS,
    linkopts = LINKOPTS,
    deps = [
        ":slam2d",
        "@boom",
        "@gtest",
    ],
)

cc_test(
    name = "region_test",
    testonly = True,
    srcs = ["region_test.cc"],
    copts = COPTS,
    linkopts = LINKOPTS,
    deps = [
        ":slam2d",
        "@boom",
        "@gtest",
    ],
)

genrule(
    name = "region_test_dsym",
    testonly = True,
    srcs = [":region_test"],
    outs = ["region_test.dSYM"],
    cmd = "dsymutil $(location :region_test) -o $@",
    output_to_bindir = True,
)

cc_test(
    name = "screen2d_test",
    testonly = True,
    srcs = ["screen2d_test.cc"],
    copts = COPTS,
    linkopts = LINKOPTS,
    deps = [
        ":slam2d",
        "@boom",
        "@gtest",
    ],
)

genrule(
    name = "screen2d_test_dsym",
    testonly = True,
    srcs = [":screen2d_test"],
    outs = ["screen2d_test.dSYM"],
    cmd = "dsymutil $(location :screen2d_test) -o $@",
    output_to_bindir = True,
)
