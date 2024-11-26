//
// Created by Bruno Schilling on 10/28/24.
// Ported from https://github.com/Visual-Computing/DynamicExplorationGraph/tree/cb7243f7296ef4513b8a5177773a7f30826c5f7b/java/deg-visualization/src/main/java/com/vc/deg/viz/om
//

#ifndef FAST_LINEAR_ASSIGNMENT_SORTER_H
#define FAST_LINEAR_ASSIGNMENT_SORTER_H

#include "map_place.hpp"

class FlasSettings {
public:
  FlasSettings(bool do_wrap, float initial_radius_factor, float radius_decay, int num_filters, float radius_end,
    float weight_swappable, float weight_non_swappable, float weight_hole, float sample_factor, int max_swap_positions)
    : do_wrap(do_wrap),
      initial_radius_factor(initial_radius_factor),
      radius_decay(radius_decay),
      num_filters(num_filters),
      radius_end(radius_end),
      weight_swappable(weight_swappable),
      weight_non_swappable(weight_non_swappable),
      weight_hole(weight_hole),
      sample_factor(sample_factor),
      max_swap_positions(max_swap_positions) {
  }

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
};

FlasSettings default_settings();
void do_sorting_full(MapPlace *map_places, int dim, int columns, int rows, const FlasSettings *settings);

#endif //FAST_LINEAR_ASSIGNMENT_SORTER_H
