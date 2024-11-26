#include "feature_data.hpp"

#include <stdlib.h>
#include <stdio.h>

// #include "det_random.h"

/*
shuffled values of a grid in a 2d plane.

shuffled with indices:
  10  5  9  1
   8 11  0 12
   2 13  7  4
  15  6 14  3
inverse indices:
   6  3  8 15
  11  1 13 10
   4  2  0  5
   7  9 14 12
*/
float *plane_2d(void) {
  const int n_cells = 16;
  const int dim = 3;
  float *features = malloc(n_cells * dim * sizeof(float));
  if (features == NULL) {
    fprintf(stderr, "Failed to allocate features.\n");
    exit(1);
  }

  const float data_features[16][3] = {
    {0.55f, 0.65f, 0.75f},
    {0.4f, 0.45f, 0.5f},
    {0.55f, 0.45f, 0.6f},
    {0.25f, 0.45f, 0.4f},
    {0.55f, 0.25f, 0.45f},
    {0.55f, 0.85f, 0.9f},
    {0.25f, 0.25f, 0.25f},
    {0.7f, 0.25f, 0.55f},
    {0.25f, 0.65f, 0.55f},
    {0.7f, 0.45f, 0.7f},
    {0.4f, 0.85f, 0.8f},
    {0.4f, 0.25f, 0.35f},
    {0.7f, 0.85f, 1.0f},
    {0.4f, 0.65f, 0.65f},
    {0.7f, 0.65f, 0.85f},
    {0.25f, 0.85f, 0.7f},
  };
  for (int i = 0; i < n_cells; i++) {
    for (int d = 0; d < dim; d++) {
      features[i * dim + d] = data_features[i][d];
    }
  }
  return features;
}

float *random_features(const int grid_size) {
  const int dim = 3;
  float *features = malloc(grid_size * grid_size * dim * sizeof(float));
  if (features == NULL) {
    fprintf(stderr, "Failed to allocate features.\n");
    exit(1);
  }
  for (int i = 0; i < grid_size * grid_size; i++) {
    for (int d = 0; d < dim; d++) {
      // features[i * dim + d] = det_next_float();
      features[i * dim + d] = rand() / (float) RAND_MAX;
    }
  }
  return features;
}
