//
// Created by Bruno Schilling on 10/28/24.
// Ported from https://github.com/Visual-Computing/DynamicExplorationGraph/tree/cb7243f7296ef4513b8a5177773a7f30826c5f7b/java/deg-visualization/src/main/java/com/vc/deg/viz/om
//

#ifndef MAP_PLACE_H
#define MAP_PLACE_H

typedef struct {
  int id;
  const float *feature;
  bool is_swappable;
} MapPlace;

void init_map_place(MapPlace *map_place, int id, const float *feature, bool is_swappable);

void init_invalid_map_place(MapPlace *map_place, const bool is_swappable);

#endif //MAP_PLACE_H
