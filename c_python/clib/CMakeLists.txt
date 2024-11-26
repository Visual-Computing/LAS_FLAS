cmake_minimum_required(VERSION 3.19)

# Only interpret if() arguments as variables or keywords when unquoted. More
# details run "cmake --help-policy CMP0054"
cmake_policy(SET CMP0054 NEW)

project(
  flas_c
  HOMEPAGE_URL "https://github.com/Visual-Computing/LAS_FLAS"
  DESCRIPTION
  "Algorithm to sort a given number of vectors on a 2-dimensional grid using a linear-assignment solver."
  VERSION 0.0.1
  LANGUAGES CXX
  )

# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(compile-options INTERFACE)
target_compile_features(compile-options INTERFACE cxx_std_20)
target_compile_definitions(compile-options INTERFACE $<$<PLATFORM_ID:Windows>:NOMINMAX>)

# setup compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
target_compile_options(
  compile-options
  INTERFACE -O3
			-march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
target_compile_options(
  compile-options
  INTERFACE -O2
			-DNDEBUG
			-march=native)
elseif(MSVC)
  # warning level 4 (show all)
  target_compile_options(compile-options INTERFACE /W4)
  
  # disable string optimizations and function level linking
  # https://stackoverflow.com/questions/5063334/what-is-the-difference-between-the-ox-and-o2-compiler-options
  target_compile_options(compile-options INTERFACE $<$<CONFIG:Release>:/Ox>)
else()
  message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

message("C++ compiler flags: ${CMAKE_CXX_FLAGS}")
message("C++ compiler flags release: ${CMAKE_CXX_FLAGS_RELEASE}")

add_library(flas_c INTERFACE
        src/feature_data.cpp
        src/feature_data.hpp
        src/flas_adapter.hpp
        src/grid_map.cpp
        src/grid_map.hpp
        src/flas_adapter.cpp
        src/fast_linear_assignment_sorter.hpp
        src/fast_linear_assignment_sorter.cpp
        src/map_place.hpp
        src/map_place.cpp
        src/junker_volgenant_solver.hpp
        src/junker_volgenant_solver.cpp
        src/det_random.cpp
        src/det_random.hpp)

target_include_directories(flas_c INTERFACE ${PROJECT_SOURCE_DIR}/src)
target_compile_features(flas_c INTERFACE cxx_std_20)
