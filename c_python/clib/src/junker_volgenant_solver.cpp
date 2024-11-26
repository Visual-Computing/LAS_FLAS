//
// Created by Bruno Schilling on 10/29/24.
// Ported from https://github.com/Visual-Computing/DynamicExplorationGraph/tree/cb7243f7296ef4513b8a5177773a7f30826c5f7b/java/deg-visualization/src/main/java/com/vc/deg/viz/om
//

#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

int *compute_assignment(int *matrix, int dim) {
  int i, imin, i0, freerow;
  int j, j1, j2 = 0, endofpath = 0, last = 0, min = 0;

  int *in_row = (int *) calloc(dim, sizeof(int));
  if (in_row == NULL) {
    fprintf(stderr, "Failed to allocate inRow.\n");
    exit(1);
  }

  int *in_col = (int *) calloc(dim, sizeof(int));
  if (in_col == NULL) {
    fprintf(stderr, "Failed to allocate inCol.\n");
    exit(1);
  }

  int *v = (int *) calloc(dim, sizeof(int));
  if (v == NULL) {
    fprintf(stderr, "Failed to allocate v.\n");
    exit(1);
  }

  int *free_ = (int *) calloc(dim, sizeof(int));
  if (free_ == NULL) {
    fprintf(stderr, "Failed to allocate free_.\n");
    exit(1);
  }

  int *collist = (int *) calloc(dim, sizeof(int));
  if (collist == NULL) {
    fprintf(stderr, "Failed to allocate collist.\n");
    exit(1);
  }

  int *matches = (int *) calloc(dim, sizeof(int));
  if (matches == NULL) {
    fprintf(stderr, "Failed to allocate matches.\n");
    exit(1);
  }

  int *pred = (int *) calloc(dim, sizeof(int));
  if (pred == NULL) {
    fprintf(stderr, "Failed to allocate pred.\n");
    exit(1);
  }

  int *d = (int *) calloc(dim, sizeof(int));
  if (d == NULL) {
    fprintf(stderr, "Failed to allocate d.\n");
    exit(1);
  }

  // skipping L53-54
  for (j = dim - 1; j >= 0; j--) {
    min = matrix[0 * dim + j];
    imin = 0;
    for (i = 1; i < dim; i++) {
      if (matrix[i * dim + j] < min) {
        min = matrix[i * dim + j];
        imin = i;
      }
    }

    v[j] = min;
    matches[imin]++;
    if (matches[imin] == 1) {
      in_row[imin] = j;
      in_col[j] = imin;
    } else {
      in_col[j] = -1;
    }
  }

  int num_free = 0;
  for (i = 0; i < dim; i++) {
    if (matches[i] == 0) {
      free_[num_free] = i;
      num_free++;
    } else if (matches[i] == 1) {
      j1 = in_row[i];
      min = INT_MAX;
      for (j = 0; j < dim; j++) {
        if (j != j1 && matrix[i * dim + j] - v[j] < min) {
          min = matrix[i * dim + j] - v[j];
        }
      }
      v[j1] -= min;
    }
  }

  for (int loop_cmt = 0; loop_cmt < 2; loop_cmt++) {
    int k = 0;
    int prv_num_free = num_free;
    num_free = 0;
    while (k < prv_num_free) {
      i = free_[k];
      k++;
      int umin = matrix[i * dim + 0] - v[0];
      j1 = 0;
      int usubmin = INT_MAX;

      for (j = 1; j < dim; j++) {
        int h = matrix[i * dim + j] - v[j];

        if (h < usubmin) {
          if (h >= umin) {
            usubmin = h;
            j2 = j;
          } else {
            usubmin = umin;
            umin = h;
            j2 = j1;
            j1 = j;
          }
        }
      }

      i0 = in_col[j1];
      if (umin < usubmin) {
        v[j1] = v[j1] - (usubmin - umin);
      } else if (i0 >= 0) {
        j1 = j2;
        i0 = in_col[j2];
      }

      in_row[i] = j1;
      in_col[j1] = i;
      if (i0 >= 0) {
        if (umin < usubmin) {
          k--;
          free_[k] = i0;
        } else {
          free_[num_free] = i0;
          num_free++;
        }
      }
    }
  }

  for (int f = 0; f < num_free; f++) {
    freerow = free_[f];
    for (j = 0; j < dim; j++) {
      d[j] = matrix[freerow * dim + j] - v[j];
      pred[j] = freerow;
      collist[j] = j;
    }

    int low = 0;
    int up = 0;
    int unassigned_found = 0;

    while (!unassigned_found) {
      if (up == low) {
        last = low - 1;
        min = d[collist[up]];
        up++;

        for (int k = up; k < dim; k++) {
          j = collist[k];
          int h = d[j];
          if (h <= min) {
            if (h < min) {
              up = low;
              min = h;
            }
            collist[k] = collist[up];
            collist[up] = j;
            up++;
          }
        }

        for (int k = low; k < up; k++) {
          if (in_col[collist[k]] < 0) {
            endofpath = collist[k];
            unassigned_found = 1;
            break;
          }
        }
      }

      if (!unassigned_found) {
        j1 = collist[low];
        low++;
        i = in_col[j1];
        int h = matrix[i * dim + j1] - v[j1] - min;

        for (int k = up; k < dim; k++) {
          j = collist[k];
          int v2 = matrix[i * dim + j] - v[j] - h;

          if (v2 < d[j]) {
            pred[j] = i;

            if (v2 == min) {
              if (in_col[j] < 0) {
                endofpath = j;
                unassigned_found = 1;
                break;
              } else {
                collist[k] = collist[up];
                collist[up] = j;
                up++;
              }
            }

            d[j] = v2;
          }
        }
      }
    }

    for (int k = 0; k <= last; k++) {
      j1 = collist[k];
      v[j1] += d[j1] - min;
    }

    i = freerow + 1;
    while (i != freerow) {
      i = pred[endofpath];
      in_col[endofpath] = i;
      j1 = endofpath;
      endofpath = in_row[i];
      in_row[i] = j1;
    }
  }

  free(in_col);
  free(v);
  free(free_);
  free(collist);
  free(matches);
  free(pred);
  free(d);

  return in_row;
}
