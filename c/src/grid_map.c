//
// Created by Bruno Schilling on 10/24/24.
// Ported from https://github.com/Visual-Computing/DynamicExplorationGraph/tree/cb7243f7296ef4513b8a5177773a7f30826c5f7b/java/deg-visualization/src/main/java/com/vc/deg/viz/om
//

#include "grid_map.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// #include "det_random.h"

// Function to create an empty grid filled with -1 values
int *empty_grid(int rows, int columns) {
  int *cells = (int *) malloc(rows * columns * sizeof(int));
  if (cells == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  for (int i = 0; i < rows * columns; i++) {
    cells[i] = i;
  }
  return cells;
}

// Function to initialize a new GridMap
GridMap init_grid_map(int rows, int columns) {
  GridMap grid;
  grid.rows = rows;
  grid.columns = columns;
  grid.cells = empty_grid(rows, columns);
  return grid;
}

// Function to calculate the total number of cells
int size(const GridMap *grid) {
  return grid->rows * grid->columns;
}

// Function to get the content of a cell by (x, y) coordinates
int get(const GridMap *grid, int x, int y) {
  return grid->cells[y * grid->columns + x];
}

// Function to get the content of a cell by index
int get_by_index(const GridMap *grid, int index) {
  return grid->cells[index];
}

// Function to set the content of a cell by (x, y) coordinates
void set(const GridMap *grid, int x, int y, int content) {
  grid->cells[y * grid->columns + x] = content;
}

// Function to set the content of a cell by index
void set_by_index(const GridMap *grid, int index, int content) {
  grid->cells[index] = content;
}

// Function to check if the grid is empty (all cells contain -1)
bool is_empty(const GridMap *grid) {
  for (int i = 0; i < size(grid); i++) {
    if (get_by_index(grid, i) != -1) {
      return false;
    }
  }
  return true;
}

// Function to check if a specific cell is empty
bool is_cell_empty(const GridMap *grid, int x, int y) {
  return get(grid, x, y) == -1;
}

// Function to count the number of free cells (cells containing -1)
int free_count(const GridMap *grid) {
  int free_count = 0;
  for (int i = 0; i < size(grid); i++) {
    if (get_by_index(grid, i) == -1) {
      free_count++;
    }
  }
  return free_count;
}

// Function to shuffle the grid cells
void shuffle(const GridMap *grid) {
  int len = size(grid);
  for (int i = 0; i < len; i++) {
    // int random_index_to_swap = det_next_int(len);
    int random_index_to_swap = rand() % len;
    int temp = grid->cells[random_index_to_swap];
    grid->cells[random_index_to_swap] = grid->cells[i];
    grid->cells[i] = temp;
  }
}

// Function to clear the grid (set all cells to -1)
void clear(const GridMap *grid) {
  for (int i = 0; i < size(grid); i++) {
    grid->cells[i] = -1;
  }
}

// Function to create a copy of the grid
GridMap *copy(const GridMap *grid) {
  GridMap *new_grid = (GridMap *) malloc(sizeof(GridMap));
  if (new_grid == NULL) {
    fprintf(stderr, "Failed to allocate new_grid.\n");
    exit(1);
  }
  new_grid->rows = grid->rows;
  new_grid->columns = grid->columns;
  new_grid->cells = (int *) malloc(size(grid) * sizeof(int));
  if (new_grid->cells == NULL) {
    fprintf(stderr, "Failed to allocate grid cells.\n");
    exit(1);
  }
  memcpy(new_grid->cells, grid->cells, size(grid) * sizeof(int));
  return new_grid;
}

// Function to print the grid information
void print_grid(const GridMap *grid) {
  printf("Grid with columns: %d, rows: %d\n", grid->columns, grid->rows);
}

void print_grid_content(const GridMap *grid) {
  for (int y = 0; y < grid->rows; y++) {
    for (int x = 0; x < grid->rows; x++) {
      printf("%d ", grid->cells[x + y * grid->columns]);
    }
    printf("\n");
  }
}

// Free the memory of a GridMap
void free_grid_map(const GridMap *grid) {
  free(grid->cells);
}