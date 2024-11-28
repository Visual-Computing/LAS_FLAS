#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "fast_linear_assignment_sorter.hpp"
#include "flas_adapter.hpp"

namespace py = pybind11;

std::tuple<int, py::array_t<int32_t> > flas_2d_features(
  const py::array_t<float, py::array::c_style> &features,
  const py::array_t<bool, py::array::c_style> &frozen,
  const bool wrap, float radius_decay, float weight_swappable, float weight_non_swappable, float weight_hole,
  int max_swap_positions
) {
  const py::buffer_info features_info = features.request();
  const ssize_t height = features_info.shape[0];
  const ssize_t width = features_info.shape[1];
  const ssize_t dim = features_info.shape[2];

  const py::array_t<int32_t> result_indices({height, width});
  const py::buffer_info result_indices_info = result_indices.request();

  if (features_info.ndim != 3)
    return std::make_tuple(1, result_indices);

  const py::buffer_info frozen_info = frozen.request();
  if (frozen_info.ndim != 2)
    return std::make_tuple(2, result_indices);

  const GridMap grid_map = init_grid_map(static_cast<int>(height), static_cast<int>(width));

  const FlasSettings settings(
    wrap, 0.5f, radius_decay, 1, 1.0f, weight_swappable, weight_non_swappable, weight_hole, 1.0f, max_swap_positions
  );

  arrange_with_holes(
    static_cast<const float *>(features_info.ptr),
    static_cast<int>(dim),
    &grid_map,
    static_cast<const bool *>(frozen_info.ptr),
    &settings
  );

  const auto res_ptr = static_cast<int32_t *>(result_indices_info.ptr);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      res_ptr[y * width + x] = get(&grid_map, x, y);
    }
  }

  return std::make_tuple(0, result_indices);
}

std::tuple<uint32_t, uint32_t> get_size(const ssize_t n_features, const float aspect_ratio) {
	uint32_t height = static_cast<uint32_t>(std::sqrt(static_cast<float>(n_features) / aspect_ratio));
	uint32_t width = static_cast<uint32_t>(static_cast<float>(height) * aspect_ratio);

	// Adjust height and width to cover all features
	while (height * width < n_features) {
		if (static_cast<float>(width) / static_cast<float>(height) < aspect_ratio) {
			width++;
		} else {
			height++;
		}
	}

	return std::make_tuple(height, width);
}

std::tuple<int, py::array_t<int32_t> > flas_1d_features(
  const py::array_t<float, py::array::c_style> &features,
  const float aspect_ratio, const bool freeze_holes, const bool wrap, float radius_decay, float weight_swappable, float weight_non_swappable,
  float weight_hole, int max_swap_positions
) {
  const py::buffer_info features_info = features.request();
  const ssize_t n_features = features_info.shape[0];
  const ssize_t dim = features_info.shape[1];
  auto [ height, width ] = get_size(n_features, aspect_ratio);

  const py::array_t<int32_t> result_indices({height, width});
  const py::buffer_info result_indices_info = result_indices.request();

  if (features_info.ndim != 2)
    return std::make_tuple(1, result_indices);

  const GridMap grid_map = init_grid_map_with_n_features(static_cast<int>(height), static_cast<int>(width), n_features);

  const FlasSettings settings(
    wrap, 0.5f, radius_decay, 1, 1.0f, weight_swappable, weight_non_swappable, weight_hole, 1.0f, max_swap_positions
  );

  bool *frozen = static_cast<bool*>(malloc(width * height * sizeof(bool)));
  for (int i = 0; i < n_features; i++) {
    frozen[i] = false;
  }
  for (int i = n_features; i < width * height; i++) {
    frozen[i] = freeze_holes;
  }

  arrange_with_holes(
    static_cast<const float *>(features_info.ptr),
    static_cast<int>(dim),
    &grid_map,
    frozen,
    &settings
  );

  free(frozen);

  const auto res_ptr = static_cast<int32_t *>(result_indices_info.ptr);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      res_ptr[y * width + x] = get(&grid_map, x, y);
    }
  }

  return std::make_tuple(0, result_indices);
}


std::tuple<uint32_t, uint32_t> get_optimal_grid_size(
  uint32_t total_num_features, float aspect_ratio, uint32_t min_height, uint32_t min_width
) {
    while (total_num_features > min_height * min_width) {
        float asp = static_cast<float>(min_width) / static_cast<float>(min_height);
        if (asp == aspect_ratio) {
          if (min_width < min_height)
            min_width++;
          else
            min_height++;
        } else if (asp < aspect_ratio) {
          min_width++;
        } else {
            min_height++;
        }
    }
    return std::make_tuple(min_height, min_width);
}

PYBIND11_MODULE(flas_c_py, m) {
  m.def("flas_2d_features", &flas_2d_features);
  m.def("flas_1d_features", &flas_1d_features);
  m.def("get_optimal_grid_size", &get_optimal_grid_size);
}
