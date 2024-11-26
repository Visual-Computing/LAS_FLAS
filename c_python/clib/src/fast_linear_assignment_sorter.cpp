//
// Created by Bruno Schilling on 10/28/24.
// Ported from https://github.com/Visual-Computing/DynamicExplorationGraph/tree/cb7243f7296ef4513b8a5177773a7f30826c5f7b/java/deg-visualization/src/main/java/com/vc/deg/viz/om
//
#include <stdbool.h>

#include "fast_linear_assignment_sorter.hpp"
#include "junker_volgenant_solver.hpp"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// #include "det_random.h"

const int QUANT = 256;

typedef struct {
  /**
   * Number of columns in the grid to sort.
   */
  int columns;

  /**
   * Number of rows in the grid to sort.
   */
  int rows;

  /**
   * grid_size = columns * rows
   */
  int grid_size;

  /**
   * Dimensionality of feature vectors.
   */
  int dim;

  /**
   * Number of positions to swap per iteration.
   */
  int num_swap_positions;

  /**
   * Array of MapPlace with size [grid_size]
   */
  MapPlace *map_places;

  /**
   * Float array with size [grid_size, dim]
   */
  float *som;

  /**
   * Float array with size [grid_size]
   */
  float *weights;

  /**
   * Current positions to swap.
   * Integer array with size [num_swap_positions];
   */
  int *swap_positions;

  /**
   * Two-dimensional float array with size [num_swap_positions, dim].
   */
  const float **fvs;

  /**
   * Two-dimensional float array with size [num_swap_positions, dim].
   */
  const float **som_fvs;

  /**
   * Array with size [num_swap_positions].
   */
  MapPlace *swapped_elements;

  /**
   * Integer distance matrix with size [num_swap_position * num_swap_positions].
   */
  int *dist_lut;

  /**
   * Distance matrix with size [num_swap_positions * num_swap_positions].
   */
  float *dist_lut_f;
} InternalData;

int min(const int a, const int b) {
  return a < b ? a : b;
}

int max(const int a, const int b) {
  return a > b ? a : b;
}

InternalData create_internal_data(MapPlace *map_places, int columns, int rows, int dim, int max_swap_positions) {
  InternalData data;

  data.columns = columns;
  data.rows = rows;
  data.grid_size = columns * rows;
  data.dim = dim;

  data.map_places = map_places;

  data.som = calloc(data.grid_size * dim, sizeof(float));
  if (data.som == NULL) {
    fprintf(stderr, "Failed to allocate som.\n");
    exit(1);
  }

  data.weights = malloc(data.grid_size * sizeof(float));
  if (data.weights == NULL) {
    fprintf(stderr, "Failed to allocate weights.\n");
    exit(1);
  }

  data.num_swap_positions = min(max_swap_positions, data.grid_size);

  data.swap_positions = malloc(data.num_swap_positions * sizeof(int));
  if (data.swap_positions == NULL) {
    fprintf(stderr, "Failed to allocate swap_positions.\n");
    exit(1);
  }

  data.fvs = malloc(data.num_swap_positions * sizeof(float *));
  if (data.fvs == NULL) {
    fprintf(stderr, "Failed to allocate fvs.\n");
    exit(1);
  }

  data.som_fvs = malloc(data.num_swap_positions * sizeof(float *));
  if (data.som_fvs == NULL) {
    fprintf(stderr, "Failed to allocate som_fvs.\n");
    exit(1);
  }

  data.swapped_elements = malloc(data.num_swap_positions * sizeof(MapPlace));
  if (data.swapped_elements == NULL) {
    fprintf(stderr, "Failed to allocate swapped_elements.\n");
    exit(1);
  }
  memcpy(data.swapped_elements, data.map_places, data.num_swap_positions * sizeof(MapPlace));

  data.dist_lut = malloc(data.num_swap_positions * data.num_swap_positions * sizeof(int));
  if (data.dist_lut == NULL) {
    fprintf(stderr, "Failed to allocate dist_lut.\n");
    exit(1);
  }

  data.dist_lut_f = malloc(data.num_swap_positions * data.num_swap_positions * sizeof(float));
  if (data.dist_lut_f == NULL) {
    fprintf(stderr, "Failed to allocate dist_lut_f.\n");
    exit(1);
  }
  return data;
}

void free_internal_data(const InternalData *data) {
  free(data->som);
  free(data->weights);
  free(data->swap_positions);
  free(data->fvs);
  free(data->som_fvs);
  free(data->swapped_elements);
  free(data->dist_lut);
  free(data->dist_lut_f);
}

void copy_feature_vectors_to_som(const InternalData *data, const FlasSettings *settings);
void filter_weighted_som(int radius_x, int radius_y, const InternalData *data, bool do_wrap);
void check_random_swaps(const InternalData *data, int radius, float sample_factor, bool do_wrap);
void filter_h_wrap(float *input, float *output, int rows, int columns, int dims, int filter_size);
void filter_h_wrap_1d(const float *input, float *output, int rows, int columns, int filter_size);
void filter_v_wrap(float *input, float *output, int rows, int columns, int dims, int filter_size);
void filter_v_wrap_1d(const float *input, float *output, int rows, int columns, int filter_size);
void filter_h_mirror(float *input, float *output, int rows, int columns, int dims, int filter_size);
void filter_h_mirror_1d(const float *input, float *output, int rows, int columns, int filter_size);
void filter_v_mirror(float *input, float *output, int rows, int columns, int dims, int filter_size);
void filter_v_mirror_1d(const float *input, float *output, int rows, int columns, int filter_size);


FlasSettings default_settings(void) {
  FlasSettings p;
  p.initial_radius_factor = 0.5f;
  p.radius_decay = 0.93f;
  p.num_filters = 1;
  p.radius_end = 1.0f;
  p.weight_swappable = 1.0f;
  p.weight_non_swappable = 100.f;
  p.weight_hole = 0.01f;
  p.sample_factor = 1.f;
  p.do_wrap = false;
  p.max_swap_positions = 9;
  return p;
}

void do_sorting(MapPlace *map_places, int dim, int columns, int rows) {
  const FlasSettings settings = default_settings();
  do_sorting_full(map_places, dim, columns, rows, &settings);
}

/**
 *
 * @param map_places Array of MapPlaces with length columns * rows. If a map place has id == -1, it is not used for
 *									 swapping.
 * @param dim The dimensionality of the features
 * @param columns Number of columns in the grid to sort
 * @param rows Number of rows in the grid to sort
 * @param settings The settings of the sorting algorithm
 */
void do_sorting_full(
  MapPlace *map_places, int dim, int columns, int rows, const FlasSettings *settings
) {
  InternalData data = create_internal_data(map_places, columns, rows, dim, settings->max_swap_positions);

  // set up the initial radius
  float rad = (float) max(columns, rows) * settings->initial_radius_factor;

  // try to improve the map
  do {
    copy_feature_vectors_to_som(&data, settings);

    int radius = max(1, (int) round(rad)); // set the radius
    int radius_x = max(1, min(columns / 2, radius));
    int radius_y = max(1, min(rows / 2, radius));
    rad *= settings->radius_decay;

    for (int i = 0; i < settings->num_filters; i++)
      filter_weighted_som(radius_x, radius_y, &data, settings->do_wrap);

    check_random_swaps(&data, radius, settings->sample_factor, settings->do_wrap);
  } while (rad > settings->radius_end);

  free_internal_data(&data);
}

void copy_feature_vectors_to_som(const InternalData *data, const FlasSettings *settings) {
  for (int pos = 0; pos < data->grid_size; pos++) {
    float *som_cell = data->som + (pos * data->dim);
    const MapPlace *cell = data->map_places + pos;

    // handle holes
    if (cell->id > -1) {
      const float *const fv = cell->feature;

      // higher weight for fixed images
      float w = cell->is_swappable ? settings->weight_swappable : settings->weight_non_swappable;
      for (int i = 0; i < data->dim; i++)
        som_cell[i] = w * fv[i];
      data->weights[pos] = w;
    } else {
      for (int i = 0; i < data->dim; i++)
        som_cell[i] *= settings->weight_hole;
      data->weights[pos] = settings->weight_hole;
    }
  }
}

void filter_weighted_som(
  int act_radius_x, int act_radius_y, const InternalData *data, bool do_wrap
) {
  int filter_size_x = 2 * act_radius_x + 1;
  int filter_size_y = 2 * act_radius_y + 1;

  float *som_h = malloc(data->grid_size * data->dim * sizeof(float));
  if (som_h == NULL) {
    fprintf(stderr, "Failed to allocate som_h.\n");
    exit(1);
  }
  float *weights_h = malloc(data->grid_size * sizeof(float));
  if (weights_h == NULL) {
    fprintf(stderr, "Failed to allocate weights_h.\n");
    exit(1);
  }

  if (do_wrap) {
    filter_h_wrap(data->som, som_h, data->rows, data->columns, data->dim, filter_size_x);
    filter_h_wrap_1d(data->weights, weights_h, data->rows, data->columns, filter_size_x);

    filter_v_wrap(som_h, data->som, data->rows, data->columns, data->dim, filter_size_y);
    filter_v_wrap_1d(weights_h, data->weights, data->rows, data->columns, filter_size_y);
  } else {
    filter_h_mirror(data->som, som_h, data->rows, data->columns, data->dim, filter_size_x);
    filter_h_mirror_1d(data->weights, weights_h, data->rows, data->columns, filter_size_x);

    filter_v_mirror(som_h, data->som, data->rows, data->columns, data->dim, filter_size_y);
    filter_v_mirror_1d(weights_h, data->weights, data->rows, data->columns, filter_size_y);
  }
  free(som_h);
  free(weights_h);

  for (int i = 0; i < data->grid_size; i++) {
    float w = 1.f / data->weights[i];
    for (int d = 0; d < data->dim; d++) {
      *(data->som + i * data->dim + d) *= w;
    }
  }
}

void shuffle_array(int *array, int size) {
  for (int i = size - 1; i > 0; i--) {
    // int index = det_next_int(i + 1);
    int index = rand() % (i+1);
    int temp = array[index];
    array[index] = array[i];
    array[i] = temp;
  }
}

void find_swap_positions_wrap(const InternalData *data, const int *swap_indices, const int num_swap_indices) {
  int start_index = (num_swap_indices - data->num_swap_positions > 0) ?
                      // ? det_next_int(num_swap_indices - data->num_swap_positions)
                      rand() % (num_swap_indices - data->num_swap_positions)
                      : 0;
  // int pos0 = det_next_int(data->grid_size);
  int pos0 = rand() % (data->rows * data->columns);

  int current_num_swap_positions = 0;
  for (int j = start_index; j < num_swap_indices && current_num_swap_positions < data->num_swap_positions; j++) {
    int d = pos0 + swap_indices[j];
    int x = d % data->columns;
    int y = (d / data->columns) % data->rows;
    int pos = y * data->columns + x;

    if (data->map_places[pos].id > -1 || data->map_places[pos].is_swappable)
      data->swap_positions[current_num_swap_positions++] = pos;
  }
}

float get_l2_distance(const float *fv1, const float *fv2, int dim) {
  float dist = 0;
  for (int i = 0; i < dim; i++) {
    float d = fv1[i] - fv2[i];
    dist += d * d;
  }
  return dist;
}

void calc_dist_lut_l2_int(const InternalData *data) {
  float max = 0;
  for (int i = 0; i < data->num_swap_positions; i++)
    for (int j = 0; j < data->num_swap_positions; j++) {
      const float val = get_l2_distance(data->fvs[i], data->som_fvs[j], data->dim);
      data->dist_lut_f[i * data->num_swap_positions + j] = val;
      if (val > max)
        max = val;
    }

  for (int i = 0; i < data->num_swap_positions; i++)
    for (int j = 0; j < data->num_swap_positions; j++) {
      data->dist_lut[i * data->num_swap_positions + j] = (int) (
        QUANT * data->dist_lut_f[i * data->num_swap_positions + j] / max + 0.5f);
    }
}

void do_swaps(const InternalData *data) {
  int num_valid = 0;
  for (int i = 0; i < data->num_swap_positions; i++) {
    int swap_position = data->swap_positions[i];
    MapPlace *swapped_element = &data->map_places[swap_position];
    data->swapped_elements[i] = *swapped_element;

    // handle holes
    if (swapped_element->id > -1) {
      data->fvs[i] = swapped_element->feature;
      num_valid++;
    } else
      data->fvs[i] = &data->som[swap_position * data->dim]; // hole

    data->som_fvs[i] = &data->som[swap_position * data->dim];
  }

  if (num_valid > 0) {
    calc_dist_lut_l2_int(data);
    int *permutation = compute_assignment(data->dist_lut, data->num_swap_positions);

    for (int i = 0; i < data->num_swap_positions; i++) {
      data->map_places[data->swap_positions[permutation[i]]] = data->swapped_elements[i];
    }

    free(permutation);
  }
}

void find_swap_positions(const InternalData *data, const int *swap_indices, int num_swap_indices, int swap_area_width,
                         int swap_area_height) {
  // calculate start position of swap area
  //int pos0 = det_next_int(data->grid_size);
  int pos0 = rand() % (data->rows * data->columns);
  int x0 = pos0 % data->columns;
  int y0 = pos0 / data->columns;

  int x_start = max(0, x0 - swap_area_width / 2);
  int y_start = max(0, y0 - swap_area_height / 2);
  if (x_start + swap_area_width > data->columns)
    x_start = data->columns - swap_area_width;
  if (y_start + swap_area_height > data->rows)
    y_start = data->rows - swap_area_height;

  int start_index = num_swap_indices - data->num_swap_positions > 0 ?
                     rand() % (num_swap_indices - data->num_swap_positions)
                     /* det_next_int(num_swap_indices - data->num_swap_positions) */
                     : 0;
  int num_swap_positions = 0;
  for (int j = start_index; j < num_swap_indices && num_swap_positions < data->num_swap_positions; j++) {
    int dx = swap_indices[j] % data->columns;
    int dy = swap_indices[j] / data->columns;

    int x = (x_start + dx) % data->columns;
    int y = (y_start + dy) % data->rows;
    int pos = y * data->columns + x;

    if (data->map_places[pos].id == -1 || data->map_places[pos].is_swappable)
      data->swap_positions[num_swap_positions++] = pos;
  }
}

void check_random_swaps(const InternalData *data, int radius, float sample_factor, bool do_wrap) {
  // set swap size
  int swap_area_width = min(2 * radius + 1, data->columns);
  int swap_area_height = min(2 * radius + 1, data->rows);
  for (int k = 0; swap_area_height * swap_area_width < data->num_swap_positions; k++) {
    if ((k & 0x1) == 0) // alternate the size increase
      swap_area_width = min(swap_area_width + 1, data->columns);
    else
      swap_area_height = min(swap_area_height + 1, data->rows);
  }

  // get all positions of the actual swap region
  const int num_swap_indices = swap_area_width * swap_area_height;
  int *swap_indices = malloc(num_swap_indices * sizeof(int));
  if (swap_indices == NULL) {
    fprintf(stderr, "Failed to allocate swap_indices.\n");
    exit(1);
  }

  int i = 0;
  for (int y = 0; y < swap_area_height; y++)
    for (int x = 0; x < swap_area_width; x++)
      swap_indices[i++] = y * data->columns + x;
  shuffle_array(swap_indices, num_swap_indices);

  int num_swap_tries = (int) max(1, (sample_factor * data->grid_size / data->num_swap_positions));
  if (do_wrap) {
    for (int n = 0; n < num_swap_tries; n++) {
      find_swap_positions_wrap(data, swap_indices, num_swap_indices);
      do_swaps(data);
    }
  } else {
    for (int n = 0; n < num_swap_tries; n++) {
      find_swap_positions(data, swap_indices, num_swap_indices, swap_area_width, swap_area_height);
      do_swaps(data);
    }
  }
  free(swap_indices);
}

void filter_h_wrap(float *input, float *output, int rows, int columns, int dims, int filter_size) {
  if (columns == 1)
    return;

  int ext = filter_size / 2; // size of the border extension

  // Allocate memory for the extended row
  float **row_ext = (float **) malloc((columns + 2 * ext) * sizeof(float *));
  if (row_ext == NULL) {
    fprintf(stderr, "Failed to allocate row_ext.\n");
    exit(1);
  }

  float *tmp = (float *) malloc(dims * sizeof(float));
  if (tmp == NULL) {
    fprintf(stderr, "Failed to allocate tmp.\n");
    exit(1);
  }

  // Filter the rows
  for (int y = 0; y < rows; y++) {
    int act_row = y * columns;

    // Copy one row
    for (int i = 0; i < columns; i++)
      row_ext[i + ext] = &input[(act_row + i) * dims];

    // Wrapped extension
    for (int i = 0; i < ext; i++) {
      row_ext[ext - 1 - i] = row_ext[columns + ext - i - 1];
      row_ext[columns + ext + i] = row_ext[ext + i];
    }

    // Set temporary storage to zero
    memset(tmp, 0, dims * sizeof(float));

    // First element
    for (int i = 0; i < filter_size; i++)
      for (int d = 0; d < dims; d++)
        tmp[d] += row_ext[i][d];

    for (int d = 0; d < dims; d++)
      output[act_row * dims + d] = tmp[d] / (float) filter_size;

    // Rest of the row
    for (int i = 1; i < columns; i++) {
      int left = i - 1;
      int right = left + filter_size;

      for (int d = 0; d < dims; d++) {
        tmp[d] += row_ext[right][d] - row_ext[left][d];
        output[(act_row + i) * dims + d] = tmp[d] / (float) filter_size;
      }
    }
  }

  free(tmp);
  free(row_ext);
}

void filter_h_wrap_1d(const float *input, float *output, int rows, int columns, int filter_size) {
  if (columns == 1)
    return;

  int ext = filter_size / 2; // size of the border extension

  // Allocate memory for the extended row
  float *row_ext = (float *) malloc((columns + 2 * ext) * sizeof(float));
  if (row_ext == NULL) {
    fprintf(stderr, "Failed to allocate row_ext.\n");
    exit(1);
  }

  // Filter the rows
  for (int y = 0; y < rows; y++) {
    int act_row = y * columns;

    // Copy one row
    for (int i = 0; i < columns; i++)
      row_ext[i + ext] = input[act_row + i];

    // Wrapped extension
    for (int i = 0; i < ext; i++) {
      row_ext[ext - 1 - i] = row_ext[columns + ext - i - 1];
      row_ext[columns + ext + i] = row_ext[ext + i];
    }

    // Temporary variable for the filter sum
    float tmp = 0.0f;

    // First element
    for (int i = 0; i < filter_size; i++)
      tmp += row_ext[i];

    output[act_row] = tmp / (float) filter_size;

    // Rest of the row
    for (int i = 1; i < columns; i++) {
      int left = i - 1;
      int right = left + filter_size;
      tmp += row_ext[right] - row_ext[left];
      output[act_row + i] = tmp / (float) filter_size;
    }
  }

  // Free the extended row
  free(row_ext);
}

void filter_v_wrap(float *input, float *output, int rows, int columns, int dims, int filter_size) {
  if (rows == 1)
    return;

  int ext = filter_size / 2; // size of the border extension

  // Allocate memory for the extended column
  float **col_ext = (float **) malloc((rows + 2 * ext) * sizeof(float *));
  if (col_ext == NULL) {
    fprintf(stderr, "Failed to allocate col_ext.\n");
    exit(1);
  }
  float *tmp = (float *) malloc(dims * sizeof(float));
  if (tmp == NULL) {
    fprintf(stderr, "Failed to allocate tmp.\n");
    exit(1);
  }

  // Filter the columns
  for (int x = 0; x < columns; x++) {
    // Copy one column
    for (int i = 0; i < rows; i++)
      col_ext[i + ext] = &input[(x + i * columns) * dims];

    // Wrapped extension
    for (int i = 0; i < ext; i++) {
      col_ext[ext - 1 - i] = col_ext[rows + ext - i - 1];
      col_ext[rows + ext + i] = col_ext[ext + i];
    }

    // Set temporary storage to zero
    memset(tmp, 0, dims * sizeof(float));

    // First element
    for (int i = 0; i < filter_size; i++)
      for (int d = 0; d < dims; d++)
        tmp[d] += col_ext[i][d];

    for (int d = 0; d < dims; d++)
      output[(x * dims) + d] = tmp[d] / filter_size;

    // Rest of the column
    for (int i = 1; i < rows; i++) {
      int left = i - 1;
      int right = left + filter_size;

      for (int d = 0; d < dims; d++) {
        tmp[d] += col_ext[right][d] - col_ext[left][d];
        output[(x + i * columns) * dims + d] = tmp[d] / filter_size;
      }
    }
  }

  free(tmp);
  free(col_ext);
}

void filter_v_wrap_1d(const float *input, float *output, int rows, int columns, int filter_size) {
  if (rows == 1)
    return;

  int ext = filter_size / 2; // size of the border extension

  // Allocate memory for the extended column
  float *col_ext = (float *) malloc((rows + 2 * ext) * sizeof(float));
  if (col_ext == NULL) {
    fprintf(stderr, "Failed to allocate col_ext.\n");
    exit(1);
  }

  // Filter the columns
  for (int x = 0; x < columns; x++) {
    // Copy one column
    for (int i = 0; i < rows; i++)
      col_ext[i + ext] = input[x + i * columns];

    // Wrapped extension
    for (int i = 0; i < ext; i++) {
      col_ext[ext - 1 - i] = col_ext[rows + ext - i - 1];
      col_ext[rows + ext + i] = col_ext[ext + i];
    }

    // Temporary variable for the filter sum
    float tmp = 0.0f;

    // First element
    for (int i = 0; i < filter_size; i++)
      tmp += col_ext[i];

    output[x] = tmp / filter_size;

    // Rest of the column
    for (int i = 1; i < rows; i++) {
      int left = i - 1;
      int right = left + filter_size;
      tmp += col_ext[right] - col_ext[left];
      output[x + i * columns] = tmp / filter_size;
    }
  }

  // Free the extended column
  free(col_ext);
}

void filter_h_mirror(float *input, float *output, int rows, int columns, int dims, int filter_size) {
  if (columns == 1)
    return;

  int ext = filter_size / 2; // size of the border extension

  // Allocate memory for the extended row
  float **row_ext = (float **) malloc((columns + 2 * ext) * sizeof(float *));
  if (row_ext == NULL) {
    fprintf(stderr, "Failed to allocate row_ext.\n");
    exit(1);
  }
  // Allocate temporary storage for filter accumulation
  float *tmp = (float *) malloc(dims * sizeof(float));
  if (tmp == NULL) {
    fprintf(stderr, "Failed to allocate tmp.\n");
    exit(1);
  }

  // Filter the rows
  for (int y = 0; y < rows; y++) {
    int act_row = y * columns;

    // Copy one row
    for (int i = 0; i < columns; i++)
      row_ext[i + ext] = &input[(act_row + i) * dims];

    // Mirrored extension
    for (int i = 0; i < ext; i++) {
      row_ext[ext - 1 - i] = row_ext[ext + i + 1];
      row_ext[columns + ext + i] = row_ext[columns + ext - 2 - i];
    }

    // Set temporary storage to zero
    memset(tmp, 0, dims * sizeof(float));

    // First element
    for (int i = 0; i < filter_size; i++)
      for (int d = 0; d < dims; d++)
        tmp[d] += row_ext[i][d];

    for (int d = 0; d < dims; d++)
      output[act_row * dims + d] = tmp[d] / filter_size;

    // Rest of the row
    for (int i = 1; i < columns; i++) {
      int left = i - 1;
      int right = left + filter_size;

      for (int d = 0; d < dims; d++) {
        tmp[d] += row_ext[right][d] - row_ext[left][d];
        output[(act_row + i) * dims + d] = tmp[d] / filter_size;
      }
    }
  }

  free(tmp);
  free(row_ext);
}

void filter_h_mirror_1d(const float *input, float *output, int rows, int columns, int filter_size) {
  if (columns == 1)
    return;

  int ext = filter_size / 2; // size of the border extension

  // Allocate memory for the extended row
  float *row_ext = (float *) malloc((columns + 2 * ext) * sizeof(float));
  if (row_ext == NULL) {
    fprintf(stderr, "Failed to allocate row_ext.\n");
    exit(1);
  }

  // Filter the rows
  for (int y = 0; y < rows; y++) {
    int act_row = y * columns;

    // Copy one row
    for (int i = 0; i < columns; i++)
      row_ext[i + ext] = input[act_row + i];

    // Mirrored extension
    for (int i = 0; i < ext; i++) {
      row_ext[ext - 1 - i] = row_ext[ext + i + 1];
      row_ext[columns + ext + i] = row_ext[columns + ext - 2 - i];
    }

    // Temporary variable for the filter sum
    float tmp = 0.0f;

    // First element
    for (int i = 0; i < filter_size; i++)
      tmp += row_ext[i];

    output[act_row] = tmp / (float) filter_size;

    // Rest of the row
    for (int i = 1; i < columns; i++) {
      int left = i - 1;
      int right = left + filter_size;
      tmp += row_ext[right] - row_ext[left];
      output[act_row + i] = tmp / (float) filter_size;
    }
  }

  // Free the extended row
  free(row_ext);
}

void filter_v_mirror(float *input, float *output, int rows, int columns, int dims, int filter_size) {
  if (rows == 1)
    return;

  int ext = filter_size / 2; // size of the border extension

  // Allocate memory for the extended column
  float **col_ext = (float **) malloc((rows + 2 * ext) * sizeof(float *));
  if (col_ext == NULL) {
    fprintf(stderr, "Failed to allocate col_ext.\n");
    exit(1);
  }
  // Allocate temporary storage for filter accumulation
  float *tmp = (float *) malloc(dims * sizeof(float));
  if (tmp == NULL) {
    fprintf(stderr, "Failed to allocate tmp.\n");
    exit(1);
  }

  // Filter the columns
  for (int x = 0; x < columns; x++) {
    // Copy one column
    for (int i = 0; i < rows; i++)
      col_ext[i + ext] = &input[(x + i * columns) * dims];

    // Mirrored extension
    for (int i = 0; i < ext; i++) {
      col_ext[ext - 1 - i] = col_ext[ext + i + 1];
      col_ext[rows + ext + i] = col_ext[ext + rows - 2 - i];
    }

    // Set temporary storage to zero
    memset(tmp, 0, dims * sizeof(float));

    // First element
    for (int i = 0; i < filter_size; i++)
      for (int d = 0; d < dims; d++)
        tmp[d] += col_ext[i][d];

    for (int d = 0; d < dims; d++)
      output[x * dims + d] = tmp[d] / filter_size;

    // Rest of the column
    for (int i = 1; i < rows; i++) {
      int left = i - 1;
      int right = left + filter_size;

      for (int d = 0; d < dims; d++) {
        tmp[d] += col_ext[right][d] - col_ext[left][d];
        output[(x + i * columns) * dims + d] = tmp[d] / filter_size;
      }
    }
  }

  free(tmp);
  free(col_ext);
}

void filter_v_mirror_1d(const float *input, float *output, int rows, int columns, int filter_size) {
  if (rows == 1)
    return;

  int ext = filter_size / 2; // size of the border extension

  // Allocate memory for the extended column
  float *col_ext = (float *) malloc((rows + 2 * ext) * sizeof(float));
  if (col_ext == NULL) {
    fprintf(stderr, "Failed to allocate col_ext.\n");
    exit(1);
  }

  // Filter the columns
  for (int x = 0; x < columns; x++) {
    // Copy one column
    for (int i = 0; i < rows; i++)
      col_ext[i + ext] = input[x + i * columns];

    // Mirrored extension
    for (int i = 0; i < ext; i++) {
      col_ext[ext - 1 - i] = col_ext[ext + i + 1];
      col_ext[rows + ext + i] = col_ext[ext + rows - 2 - i];
    }

    // Temporary variable for the filter sum
    float tmp = 0.0f;

    // First element
    for (int i = 0; i < filter_size; i++)
      tmp += col_ext[i];

    output[x] = tmp / (float) filter_size;

    // Rest of the column
    for (int i = 1; i < rows; i++) {
      int left = i - 1;
      int right = left + filter_size;
      tmp += col_ext[right] - col_ext[left];
      output[x + i * columns] = tmp / (float) filter_size;
    }
  }

  // Free the extended column
  free(col_ext);
}
