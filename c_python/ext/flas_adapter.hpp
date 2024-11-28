//
// Created by Bruno Schilling on 10/24/24.
// Ported from https://github.com/Visual-Computing/DynamicExplorationGraph/tree/cb7243f7296ef4513b8a5177773a7f30826c5f7b/java/deg-visualization/src/main/java/com/vc/deg/viz/om
//

#ifndef FLAS_ADAPTER_H
#define FLAS_ADAPTER_H

#include "fast_linear_assignment_sorter.hpp"
#include "grid_map.hpp"

void arrange_with_holes(const float *features, int dim, const GridMap *map, const bool *frozen, const FlasSettings* settings);

#endif //FLAS_ADAPTER_H
