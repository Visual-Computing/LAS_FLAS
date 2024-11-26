//
// Created by Bruno Schilling on 10/28/24.
// Ported from https://github.com/Visual-Computing/DynamicExplorationGraph/tree/cb7243f7296ef4513b8a5177773a7f30826c5f7b/java/deg-visualization/src/main/java/com/vc/deg/viz/om
//

#ifndef FAST_LINEAR_ASSIGNMENT_SORTER_H
#define FAST_LINEAR_ASSIGNMENT_SORTER_H

#include "flas_adapter.hpp"
#include "map_place.hpp"

typedef struct {
  bool do_wrap;
  float initial_radius_factor;
  float radius_decay;
  int num_filters;
  float radius_end;
  float weight_swappable;
  float weight_non_swappable;
  float weight_hole;
  float sample_factor;
  int max_swap_positions;
} FlasSettings;

FlasSettings default_settings(void);
void do_sorting(MapPlace *map_places, int dim, int columns, int rows);
void do_sorting_full(MapPlace *map_places, int dim, int columns, int rows, const FlasSettings *settings);

#endif //FAST_LINEAR_ASSIGNMENT_SORTER_H
