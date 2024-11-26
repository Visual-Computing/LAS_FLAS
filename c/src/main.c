#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#include <gtk/gtk.h>

#include "feature_data.h"
#include "grid_map.h"
#include "flas_adapter.h"

const int DIM = 3;
const bool DO_WRAP = false;

bool *create_in_use(int grid_size, bool value);

void show_grid(const GridMap *grid, float *features, int argc, char **argv, int scale);

int main(int argc, char **argv) {
  srand((unsigned int)time(NULL));
  int grid_size = 32;

  // float* features = plane_2d(); grid_size = 4;
  float *features = random_features(grid_size);

  GridMap grid = init_grid_map(grid_size, grid_size);
  bool *in_use = create_in_use(grid_size, false);
  clock_t start = clock();
  arrange_with_holes(features, DIM, &grid, in_use, DO_WRAP);
  clock_t end = clock();

  // print_grid_content(&grid);

  show_grid(&grid, features, argc, argv, 8);

  free(in_use);
  free_grid_map(&grid);
  free(features);

  double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Execution time: %f seconds\n", cpu_time_used);

  return 0;
}

bool *create_in_use(const int grid_size, const bool value) {
  bool *in_use = malloc(grid_size * grid_size * sizeof(bool));
  if (in_use == NULL) {
    fprintf(stderr, "Failed to malloc in_use.\n");
    exit(1);
  }
  for (int i = 0; i < grid_size * grid_size; i++) {
    in_use[i] = value;
  }
  return in_use;
}

GtkWidget *create_image_from_pixels(const float *pixels, int source_width, int source_height, int scale) {
  int scaled_width = source_width * scale;
  int scaled_height = source_height * scale;

  // Allocate memory for pixel data in 8-bit RGB format
  guchar *rgb_data = (guchar *) malloc(scaled_width * scaled_height * 3);
  if (rgb_data == NULL) {
    fprintf(stderr, "Failed to allocate rgb_data.\n");
    exit(1);
  }

  // Convert the float pixel data to 8-bit RGB data
  for (int i = 0; i < source_width * source_height * 3; i++) {
    rgb_data[i] = (guchar) (pixels[i] * 255);
  }
  for (int y = 0; y < scaled_height; y++) {
    for (int x = 0; x < scaled_width; x++) {
      int source_x = x / scale;
      int source_y = y / scale;
      int source_index = (source_y * source_width + source_x) * 3;
      int target_index = (y * scaled_width + x) * 3;
      for (int d = 0; d < DIM; d++) {
        rgb_data[target_index + 2 - d] = (guchar) (pixels[source_index + d] * 255);
      }
    }
  }

  // Create a GdkPixbuf from the 8-bit RGB data
  GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
    rgb_data, // Pointer to pixel data
    GDK_COLORSPACE_RGB, // Color space: RGB
    FALSE, // No alpha channel
    8, // Bits per sample
    scaled_width, // Image width
    scaled_height, // Image height
    scaled_width * 3, // Row stride (number of bytes per row)
    (GdkPixbufDestroyNotify) free, // Function to free the pixel data
    NULL // User data for destroy function
  );

  // Create a GtkImage widget from the GdkPixbuf
  GtkWidget *image = gtk_image_new_from_pixbuf(pixbuf);

  // Unref the GdkPixbuf since the GtkImage holds a reference now
  g_object_unref(pixbuf);

  return image;
}

void destroy(void) {
  gtk_main_quit();
}

void show_grid(const GridMap *grid, float *features, int argc, char **argv, int scale) {
  gtk_init(&argc, &argv);

  GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  int width = grid->columns;
  int height = grid->rows;
  float *pixels = (float *) malloc(width * height * 3 * sizeof(float));
  if (pixels == NULL) {
    fprintf(stderr, "Failed to pixels.\n");
    exit(1);
  }
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int id = get(grid, x, y);
      float *feature = features + (id * DIM);
      int index = (x + y * width) * 3;
      for (int i = 0; i < DIM; i++) {
        pixels[index + i] = feature[i];
      }
    }
  }
  GtkWidget *image = create_image_from_pixels(pixels, width, height, scale);
  free(pixels);

  gtk_signal_connect(GTK_OBJECT(window), "destroy", GTK_SIGNAL_FUNC(destroy), NULL);

  gtk_container_add(GTK_CONTAINER(window), image);

  gtk_widget_show_all(window);

  gtk_main();
}
