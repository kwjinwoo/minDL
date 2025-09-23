cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(miniDL LANGUAGES CXX)

option(MINIDL_BUILD_TESTS "Build miniDL tests." ON)
option(MINIDL_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

function(minidl_set_warnings TARGET_NAME)
  if (MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE /W4)
  else()
    target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wpedantic)
    if (MINIDL_WARNINGS_AS_ERRORS)
      target_compile_options(${TARGET_NAME} PRIVATE -Werror)
    endif()
  endif()
endfunction()

add_subdirectory(src)

if(MINIDL_BUILD_TESTS)
  include(FetchContent)
  set(BUILD_GTEST ON  CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_subdirectory(tests)
endif()