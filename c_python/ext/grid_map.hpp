//
// Created by Bruno Schilling on 10/24/24.
// Ported from https://github.com/Visual-Computing/DynamicExplorationGraph/tree/cb7243f7296ef4513b8a5177773a7f30826c5f7b/java/deg-visualization/src/main/java/com/vc/deg/viz/om
//
#ifndef GRID_MAP_H
#define GRID_MAP_H

typedef struct {
  int rows;
  int columns;
  int *cells;
} GridMap;

// Function to create an empty grid filled with -1 values
int *empty_grid(int rows, int columns);

// Function to initialize a new GridMap
GridMap init_grid_map(int rows, int columns);

// Function to calculate the total number of cells
int size(const GridMap *grid);

// Function to get the content of a cell by (x, y) coordinates
int get(const GridMap *grid, int x, int y);

// Function to get the content of a cell by index
int get_by_index(const GridMap *grid, int index);

// Function to set the content of a cell by (x, y) coordinates
void set(const GridMap *grid, int x, int y, int content);

// Function to set the content of a cell by index
void set_by_index(const GridMap *grid, int index, int content);

// Function to check if the grid is empty (all cells contain -1)
bool is_empty(const GridMap *grid);

// Function to check if a specific cell is empty
bool is_cell_empty(const GridMap *grid, int x, int y);

// Function to count the number of free cells (cells containing -1)
int free_count(const GridMap *grid);

// Function to shuffle the grid cells
void shuffle(const GridMap *grid);

// Function to clear the grid (set all cells to -1)
void clear(const GridMap *grid);

// Function to create a copy of the grid
GridMap *copy(const GridMap *grid);

// Function to print the grid information
void print_grid(const GridMap *grid);

void print_grid_content(const GridMap *grid);

// Free the memory of a GridMap
void free_grid_map(const GridMap *grid);
#endif // GRID_MAP_H
