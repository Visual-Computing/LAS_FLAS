#include <iostream>
#include <pybind11/pybind11.h>

#include "flas_adapter.hpp"
#include "grid_map.hpp"

namespace py = pybind11;

bool say_hello() {
  std::cout << "hello world" << std::endl;
  return false;
}

void arrange_with_holes_impl(const float *features, const int dim, const GridMap *map, const bool *in_use, bool do_wrap) {
  arrange_with_holes(features, dim, map, in_use, do_wrap);
}

PYBIND11_MODULE(flas_c_py, m) {
  m.def("say_hello", &say_hello);
  m.def("arrange_with_holes", &arrange_with_holes_impl);

  py::class_<GridMap>(m, "GridMap");
}
