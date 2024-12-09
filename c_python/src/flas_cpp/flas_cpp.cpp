#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// ReSharper disable once CppUnusedIncludeDirective
#include <pybind11/functional.h>

#include "fast_linear_assignment_sorter.hpp"
#include "metrics.hpp"

namespace py = pybind11;

std::tuple<int, py::array_t<int32_t> > flas(
  const py::array_t<float, py::array::c_style> &features,
  const py::array_t<int32_t, py::array::c_style> &ids,
  const py::array_t<bool, py::array::c_style> &frozen,
  const bool wrap, float radius_decay, float weight_swappable, float weight_non_swappable, float weight_hole,
  int max_swap_positions, int seed, int optimize_narrow_grids, const std::function<bool(float)>& callback
) {
  // ids
  const py::buffer_info ids_info = ids.request();
  if (ids_info.ndim != 2) {
    const py::array_t<int32_t> tmp(0);
    return std::make_tuple(1, tmp);
  }
  const int height = static_cast<int>(ids_info.shape[0]);
  const int width = static_cast<int>(ids_info.shape[1]);
  const int32_t* ids_ptr = static_cast<const int32_t *>(ids_info.ptr);

  // features
  const py::buffer_info features_info = features.request();
  if (features_info.ndim != 2) {
    const py::array_t<int32_t> tmp(0);
    return std::make_tuple(2, tmp);
  }
  const int dim = static_cast<int>(features_info.shape[1]);
  const float* features_ptr = static_cast<const float *>(features_info.ptr);

  // frozen
  const py::buffer_info frozen_info = frozen.request();
  if (frozen_info.ndim != 2) {
    const py::array_t<int32_t> result_indices(0);
    return std::make_tuple(3, result_indices);
  }

  if (frozen_info.shape[0] != height || frozen_info.shape[1] != width) {
    const py::array_t<int32_t> result_indices(0);
    return std::make_tuple(4, result_indices);
  }
  const bool* frozen_ptr = static_cast<bool*>(frozen_info.ptr);

  // settings
  const FlasSettings settings(
    wrap, 0.5f, radius_decay, 1, 1.0f, weight_swappable, weight_non_swappable, weight_hole, 1.0f, max_swap_positions,
    optimize_narrow_grids
  );

  // random
  RandomEngine rng;
  if (seed == -1)
    rng.seed(std::random_device()());
  else
    rng.seed(seed);

  // create map fields
  auto map_fields = static_cast<MapField *>(malloc(height * width * sizeof(MapField)));
  if (map_fields == nullptr) {
    std::cerr << "Failed to allocate map_fields.\n" << std::endl;
    exit(1);
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const int index = x + y * static_cast<int>(width);
      int id = ids_ptr[index];
      MapField *current_field = &map_fields[index];
      const bool field_frozen = frozen_ptr[index];
      if (id != -1) {
        const float *const float_feature = features_ptr + (id * dim);
        init_map_field(current_field, id, float_feature, !field_frozen);
      } else {
        init_invalid_map_field(current_field, !field_frozen);
      }
    }
  }
  // ----------------------------------------
  do_sorting_full(
    map_fields, static_cast<int>(dim), static_cast<int>(width), static_cast<int>(height), &settings, &rng, callback
  );

  const py::array_t<int32_t> result_indices({height, width});
  const py::buffer_info result_indices_info = result_indices.request();

  const auto res_ptr = static_cast<int32_t *>(result_indices_info.ptr);

  // convert back --------------------------
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const MapField map_field = map_fields[x + y * width];
      res_ptr[y * width + x] = map_field.id;
    }
  }

  free(map_fields);
  // ---------------------------------------

  return std::make_tuple(0, result_indices);
}

std::tuple<int, py::array_t<int32_t> > flas_no_callback(
  const py::array_t<float, py::array::c_style> &features,
  const py::array_t<int32_t, py::array::c_style> &ids,
  const py::array_t<bool, py::array::c_style> &frozen,
  const bool wrap, float radius_decay, float weight_swappable, float weight_non_swappable, float weight_hole,
  int max_swap_positions, int seed, int optimize_narrow_grids
) {
  return flas(
    features, ids, frozen, wrap, radius_decay, weight_swappable, weight_non_swappable, weight_hole, max_swap_positions,
    seed, optimize_narrow_grids, [](float) { return false; }
  );
}

std::tuple<uint32_t, uint32_t> get_size(const size_t n_features, const float aspect_ratio) {
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

/**
 * @param features The features array with shape [h, w, d].
 * @param valid Boolean array defining valid (not holes) fields with shape [h, w]
 * @param wrap Whether features can be wrapped around the plane.
 * @return A tuple [error_code, num_distances, sum_distance], where
 *         - error_code indices an error, if != 0
 *         - num_distances, is the number of distances that where calculated to substitute holes
 *         - sum_distance, is the sum of all distances that where calculated to substitute holes
 */
std::tuple<unsigned int, unsigned int, double> calc_hole_substitution_distance(
  const py::array_t<double, py::array::c_style> &features,
  const py::array_t<bool, py::array::c_style> &valid,
  const bool wrap
) {
  // features
  const py::buffer_info features_info = features.request();
  if (features_info.ndim != 3) {
    return std::make_tuple(1, 0, 0.0);
  }
  const int height = static_cast<int>(features_info.shape[0]);
  const int width = static_cast<int>(features_info.shape[1]);
  const int dim = static_cast<int>(features_info.shape[2]);
  const double* features_ptr = static_cast<const double *>(features_info.ptr);

  // valid
  const py::buffer_info valid_info = valid.request();
  if (valid_info.ndim != 2) {
    return std::make_tuple(2, 0, 0.0);
  }
  if (valid_info.shape[0] != height || valid_info.shape[1] != width) {
    return std::make_tuple(3, 0, 0.0); // wrong shape
  }
  const bool* valid_ptr = static_cast<const bool *>(valid_info.ptr);

  // calculation
  auto [num_dists, sum_dists] = calc_substitution_distance(height, width, dim, wrap, features_ptr, valid_ptr);
  return std::make_tuple(0, num_dists, sum_dists);
}

/**
 * @param features The features array with shape [h, w, d].
 * @param wrap Whether features can be wrapped around the plane.
 * @return A tuple [error_code, num_distances, sum_distance], where
 *         - error_code indices an error, if != 0
 *         - num_distances, is the number of distances that where calculated to substitute holes
 *         - sum_distance, is the sum of all distances that where calculated to substitute holes
 */
std::tuple<unsigned int, unsigned int, double> calc_hole_substitution_distance_all_valid(
  const py::array_t<double, py::array::c_style> &features,
  const bool wrap
) {
  // features
  const py::buffer_info features_info = features.request();
  if (features_info.ndim != 3) {
    return std::make_tuple(1, 0, 0.0);
  }
  const int height = static_cast<int>(features_info.shape[0]);
  const int width = static_cast<int>(features_info.shape[1]);
  const int dim = static_cast<int>(features_info.shape[2]);
  const double* features_ptr = static_cast<const double *>(features_info.ptr);

  // calculation
  auto [num_dists, sum_dists] = calc_substitution_distance(height, width, dim, wrap, features_ptr, nullptr);
  return std::make_tuple(0, num_dists, sum_dists);
}

PYBIND11_MODULE(flas_cpp, m) {
  m.def("flas", &flas);
  m.def("flas_no_callback", &flas_no_callback);
  m.def("get_optimal_grid_size", &get_optimal_grid_size);
  m.def("calc_hole_substitution_distance", &calc_hole_substitution_distance);
  m.def("calc_hole_substitution_distance_all_valid", &calc_hole_substitution_distance_all_valid);
}
