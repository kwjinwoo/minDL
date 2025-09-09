cmake_minimum_required(VERSION 3.14)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(miniDL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

if(APPLE)
  set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "" FORCE)
endif()

include(FetchContent)
set(BUILD_GTEST ON  CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE MINIDL_SOURCES src/*.cpp)
add_library(minidl ${MINIDL_SOURCES})
target_include_directories(minidl PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(minidl PUBLIC cxx_std_17)

FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)
