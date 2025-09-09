file(GLOB TEST_SOURCES *.cpp)
add_executable(minidl_tests ${TEST_SOURCES})

target_link_libraries(minidl_tests PRIVATE
  GTest::gtest_main
  GTest::gtest
  minidl
)

target_compile_features(minidl_tests PRIVATE cxx_std_17)

include(GoogleTest)
gtest_discover_tests(minidl_tests)
