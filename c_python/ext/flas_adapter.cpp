//
// Created by Bruno Schilling on 10/24/24.
// Ported from https://github.com/Visual-Computing/DynamicExplorationGraph/tree/cb7243f7296ef4513b8a5177773a7f30826c5f7b/java/deg-visualization/src/main/java/com/vc/deg/viz/om
//
#include "flas_adapter.hpp"
#include "fast_linear_assignment_sorter.hpp"
#include "map_field.hpp"

#include <iostream>

void arrange_with_holes(const float *features, const int dim, const GridMap *map, const bool *frozen, const FlasSettings* settings) {
  int rows = map->rows;
  int columns = map->columns;

  // copy the data
  auto map_fields = static_cast<MapField *>(malloc(rows * columns * sizeof(MapField)));
  if (map_fields == nullptr) {
    std::cerr << "Failed to allocate map_fields.\n" << std::endl;
    exit(1);
  }
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < columns; x++) {
      int content = get(map, x, y);
      MapField *current_field = &map_fields[x + y * columns];
      const bool field_frozen = frozen[x + y * columns];
      if (content != -1) {
        const float *const float_feature = features + (content * dim);
        init_map_field(current_field, content, float_feature, !field_frozen);
      } else {
        init_invalid_map_field(current_field, !field_frozen);
      }
    }
  }

  // Sort the map
  do_sorting_full(map_fields, dim, columns, rows, settings);

  // apply the new order to the map
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < columns; x++) {
      const MapField map_field = map_fields[x + y * columns];
      if (map_field.id > -1) {
        set(map, x, y, map_field.id);
      } else {
        set(map, x, y, -1);
      }
    }
  }

  free(map_fields);
}
