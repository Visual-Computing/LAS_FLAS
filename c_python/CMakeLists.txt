# python-bindings/CMakeLists.txt
cmake_minimum_required(VERSION 3.19)
project(flas_c_py)

# Add python executable
if(DEFINED PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Add pybind11
find_package(pybind11 CONFIG REQUIRED)

# Specify the include directories
include_directories(${CMAKE_SOURCE_DIR}/clib/src)

# Add cpp-flas
add_subdirectory(${CMAKE_SOURCE_DIR}/clib/)
pybind11_add_module(flas_c_py clib/src/flas_c.cpp)

target_link_libraries(flas_c_py PRIVATE compile-options)
