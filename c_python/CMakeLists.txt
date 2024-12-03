cmake_minimum_required(VERSION 3.19)

cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0148 NEW)

# Project name and language
project(flas_cpp LANGUAGES CXX)

# Include pybind11
find_package(pybind11 CONFIG REQUIRED)

# Add library sources
set(LIBRARY_SOURCES
	ext/det_random.cpp
	ext/det_random.hpp
	ext/fast_linear_assignment_sorter.cpp
	ext/fast_linear_assignment_sorter.hpp
	ext/junker_volgenant_solver.cpp
	ext/junker_volgenant_solver.hpp
	ext/map_field.cpp
	ext/map_field.hpp
)

# Create the library (STATIC or SHARED, depending on needs)
add_library(flas_c SHARED ${LIBRARY_SOURCES})
set_target_properties(flas_c PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Include directories for the library
target_include_directories(flas_c PUBLIC
    ${CMAKE_SOURCE_DIR}/ext
)

# Add Python bindings
pybind11_add_module(flas_cpp flas/flas_c.cpp)

# Link the C++ library to the Python module
target_link_libraries(flas_cpp PRIVATE flas_c)
