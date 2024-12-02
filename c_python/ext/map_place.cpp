//
// Created by Bruno Schilling on 10/28/24.
// Ported from https://github.com/Visual-Computing/DynamicExplorationGraph/tree/cb7243f7296ef4513b8a5177773a7f30826c5f7b/java/deg-visualization/src/main/java/com/vc/deg/viz/om
//
#include "map_place.hpp"

void init_map_place(MapPlace *map_place, const int id, const float *const feature, const bool is_swappable) {
  map_place->id = id;
  map_place->feature = feature;
  map_place->is_swappable = is_swappable;
}

void init_invalid_map_place(MapPlace *map_place, const bool is_swappable) {
  map_place->id = -1;
  map_place->feature = nullptr;
  map_place->is_swappable = is_swappable;
}

int get_num_valid_map_places(const MapPlace *map_places, const int num_map_places) {
  int num_valid_map_places = 0;
  for (int i = 0; i < num_map_places; i++) {
    if (map_places[i].id != -1) {
      num_valid_map_places++;
    }
  }
  return num_valid_map_places;
}
