cmake_minimum_required(VERSION 3.19)

cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0148 NEW)

# Project name and language
project(flas_cpp LANGUAGES CXX)

if(DEFINED PYTHON_EXECUTABLE)
	set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

# Include pybind11
find_package(pybind11 CONFIG REQUIRED)

# Add library sources
set(LIBRARY_SOURCES
	ext/fast_linear_assignment_sorter.hpp
	ext/junker_volgenant_solver.hpp
	ext/map_field.hpp
)

# Create the library
add_library(flas_c INTERFACE)
target_include_directories(flas_c INTERFACE
    ${CMAKE_SOURCE_DIR}/ext
)

# Add Python bindings
pybind11_add_module(flas_cpp src/flas_cpp/flas_cpp.cpp)

# Link the C++ library to the Python module
target_link_libraries(flas_cpp PRIVATE flas_c)
