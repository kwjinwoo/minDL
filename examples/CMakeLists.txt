add_executable(mnist_mlp
  mnist_mlp.cpp
)

target_link_libraries(mnist_mlp
  PRIVATE
    minidl_core
    minidl_ops
    minidl_nn
)

target_include_directories(mnist_mlp
  PRIVATE
    ${MINIDL_PUBLIC_INCLUDE_DIR}
)