set(MINIDL_PUBLIC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

# 1) minidl_core
add_library(minidl_core STATIC
    tensor/tensor_core.cpp
    tensor/tensor_factories.cpp
    tensor/tensor_view.cpp
    detail/layout.cpp
    detail/iter.cpp
    detail/config.cpp
)
target_include_directories(minidl_core
    PUBLIC
        ${MINIDL_PUBLIC_INCLUDE_DIR}
)
minidl_set_warnings(minidl_core)

# 2) minidl_ops
add_library(minidl_ops STATIC
    ops/pointwise.cpp
    ops/matmul.cpp
    ops/reduction.cpp
    ops/loss.cpp
    kernels/kernels_pointwise.cpp
)
target_include_directories(minidl_ops
    PUBLIC
        ${MINIDL_PUBLIC_INCLUDE_DIR}
)
target_link_libraries(minidl_ops
  PUBLIC  minidl_core OpenMP::OpenMP_CXX
)
minidl_set_warnings(minidl_ops)

# 3) minidl_nn
add_library(minidl_nn STATIC
nn/module.cpp
)
target_include_directories(minidl_nn
PUBLIC
${MINIDL_PUBLIC_INCLUDE_DIR}
)
minidl_set_warnings(minidl_nn)

set_target_properties(minidl_ops minidl_nn PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# interface
add_library(minidl INTERFACE)
target_link_libraries(minidl
    INTERFACE
        minidl_core
        minidl_ops
        minidl_nn
)
target_include_directories(minidl
    INTERFACE
        ${MINIDL_PUBLIC_INCLUDE_DIR}
)