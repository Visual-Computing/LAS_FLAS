cmake_minimum_required(VERSION 3.29)
project(FLAS_C C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK REQUIRED gtk+-2.0)

include_directories(${GTK_INCLUDE_DIRS})
link_directories(${GTK_LIBRARY_DIRS})

add_executable(FLAS_C
    src/main.c
        src/feature_data.c
        src/feature_data.h
        src/flas_adapter.h
        src/grid_map.c
        src/grid_map.h
        src/flas_adapter.c
        src/fast_linear_assignment_sorter.h
        src/fast_linear_assignment_sorter.c
        src/map_place.h
        src/map_place.c
        src/junker_volgenant_solver.h
        src/junker_volgenant_solver.c
        src/det_random.c
        src/det_random.h)

target_link_libraries(FLAS_C ${GTK_LIBRARIES})
target_link_libraries(FLAS_C m)
