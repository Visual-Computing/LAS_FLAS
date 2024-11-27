#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "fast_linear_assignment_sorter.hpp"
#include "flas_adapter.hpp"

namespace py = pybind11;

std::tuple<int, py::array_t<uint32_t> > flas(
  const py::array_t<float, py::array::c_style> &features,
  const py::array_t<bool, py::array::c_style> &in_use,
  const bool wrap, const float initial_radius_factor, float radius_decay, int num_filters, float radius_end,
  float weight_swappable, float weight_non_swappable, float weight_hole, float sample_factor, int max_swap_positions
) {
  const py::buffer_info features_info = features.request();
  const ssize_t height = features_info.shape[0];
  const ssize_t width = features_info.shape[1];
  const ssize_t dim = features_info.shape[2];

  const py::array_t<uint32_t> result_indices({height, width});
  const py::buffer_info result_indices_info = result_indices.request();

  if (features_info.ndim != 3)
    return std::make_tuple(1, result_indices);

  const py::buffer_info in_use_info = in_use.request();
  if (in_use_info.ndim != 2)
    return std::make_tuple(2, result_indices);

  const GridMap grid_map = init_grid_map(static_cast<int>(height), static_cast<int>(width));

  const FlasSettings settings(wrap, initial_radius_factor, radius_decay, num_filters, radius_end, weight_swappable,
                        weight_non_swappable, weight_hole, sample_factor, max_swap_positions);

  arrange_with_holes(
    static_cast<const float *>(features_info.ptr),
    static_cast<int>(dim),
    &grid_map,
    static_cast<const bool *>(in_use_info.ptr),
    &settings
  );

  const auto res_ptr = static_cast<uint32_t *>(result_indices_info.ptr);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      res_ptr[y * width + x] = get(&grid_map, x, y);
    }
  }

  return std::make_tuple(0, result_indices);
}

PYBIND11_MODULE(flas_c_py, m) {
  m.def("flas", &flas);
}
