add_executable(bench_gemm
    gemm_bench.cpp
)

target_link_libraries(bench_gemm PRIVATE
    minidl
)
target_compile_features(bench_gemm PRIVATE cxx_std_17)
target_compile_options(bench_gemm PRIVATE -O3 -DNDEBUG -march=native)

set_target_properties(bench_gemm PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)