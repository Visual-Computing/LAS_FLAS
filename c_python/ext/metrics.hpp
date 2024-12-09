//
// Created by alok on 09.12.2024.
//

#ifndef METRICS_HPP
#define METRICS_HPP

#include <tuple>

inline bool get_valid(const bool* const valid, size_t y, size_t x, size_t height, size_t width, bool wrap) {
  if (wrap) {
    // wrap around the plane
    while (x < 0)
      x += width;
    while (y < 0)
      y += height;
    while (x >= width)
      x -= width;
    while (y >= height)
      y -= height;
  } else {
    if (x < 0 || y < 0 || x >= width || y >= height) {
      return false;
    }
  }
  if (valid == nullptr)
    return true;
  return valid[x + y * width];
}

inline const float* get_feature(
  const float* const features, size_t y, size_t x, size_t height, size_t width, size_t dim, bool wrap
) {
  if (wrap) {
    // wrap around the plane
    while (x < 0)
      x += width;
    while (y < 0)
      y += height;
    while (x >= width)
      x -= width;
    while (y >= height)
      y -= height;
  } else {
    if (x < 0 || y < 0 || x >= width || y >= height) {
      return nullptr;
    }
  }
  return &features[x * dim + y * width * dim];
}

class L1DistanceIterator {
public:
    using Point = std::tuple<int, int>;

    L1DistanceIterator(int center_y, int center_x, int distance, bool end = false)
        : center_y(center_y), center_x(center_x), distance(distance), index(end?distance*4:0)
	{}

    // Dereference operator to access the current point
    Point operator*() const {
		int _x = -distance + (index + 1) / 2;

		int _y = (index+1) / 2;
		if (_y > distance)
			_y = 2*distance - _y;
		if (index % 2 == 0)
			_y = -_y;

        return std::make_tuple(_y + center_y, _x + center_x);
    }

    // Pre-increment operator
    L1DistanceIterator& operator++() {
        ++index;
        return *this;
    }

    // Equality operator
    bool operator==(const L1DistanceIterator& other) const {
        return index == other.index;
    }

    // Inequality operator
    bool operator!=(const L1DistanceIterator& other) const {
        return !(*this == other);
    }

private:
    int center_y, center_x, distance, index;
};

class L1DistanceRange {
public:
    using iterator = L1DistanceIterator;

    L1DistanceRange(int center_y, int center_x, int distance)
        : center_y(center_y), center_x(center_x), distance(distance) {}

    [[nodiscard]] iterator begin() const {
        return {center_y, center_x, distance};
    }

    [[nodiscard]] iterator end() const {
        return {center_y, center_x, distance, true};
    }

private:
    int center_y, center_x, distance;
};

inline unsigned int num_substitutions_needed(
  size_t y, size_t x, size_t height, size_t width, const bool* const valid, bool wrap
) {
  // if hole, dont substitute
  if (!get_valid(valid, y, x, height, width, false)) {
    return 0;
  }

  unsigned int num_subs = 0;
  for (const auto& [y, x] : L1DistanceRange(static_cast<int>(y), static_cast<int>(x), 1)) {
    if (!get_valid(valid, y, x, height, width, wrap)) {
      num_subs++;
    }
  }
  return num_subs;
}

inline double get_best_substitution_partner_distances_sum(
  size_t center_y, size_t center_x, size_t height, size_t width, size_t dim, bool wrap, const float* const features,
  const bool* const valid, unsigned int num_subs
) {
  const float* center_feature = get_feature(features, center_y, center_x, height, width, dim, wrap);
  int distance = 2; // distance-1 partners are already used, so we start at 2.
  unsigned int num_found = 0;
  std::array<float, 4> best_distances{};
  while (num_found != num_subs) {
    for (const auto& [y, x] : L1DistanceRange(static_cast<int>(center_y), static_cast<int>(center_x), distance)) {
      if (get_valid(valid, y, x, height, width, wrap)) {
        const float* const feature = get_feature(features, y, x, height, width, dim, wrap);
        if (feature != nullptr) {
          float feat_distance = get_l2_distance(center_feature, feature, static_cast<int>(dim));
          if (num_found < num_subs) {
            best_distances[num_found] = feat_distance;
            num_found++;
          } else {
            // replace worst feature with current feature
            int worst_feat_index = 0;
            for (int i = 1; i < num_found; i++) {
              if (best_distances[i] > best_distances[worst_feat_index]) {
                worst_feat_index = i;
              }
            }
            if (best_distances[worst_feat_index] > feat_distance) {
              best_distances[worst_feat_index] = feat_distance;
            }
          }
        }
      }
    }
    distance++;
  }

  double sum = 0;
  for (int i = 0; i < num_subs; i++) {
    sum += best_distances[i];
  }
  return sum;
}

inline std::tuple<unsigned int, double> calc_substitution_distance(
  size_t height, size_t width, size_t dim, bool wrap, const float* const features, const bool* const valid
  ) {
  unsigned int num_dists = 0;
  double sum_dists = 0;
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      // needs substitution partners?
      unsigned int num_subs = num_substitutions_needed(y, x, height, width, valid, wrap);
      if (num_subs) {
        double dist = get_best_substitution_partner_distances_sum(
          y, x, height, width, dim, wrap, features, valid, num_subs
        );
        num_dists += num_subs;
        sum_dists += dist;
      }
    }
  }
  return std::make_tuple(num_dists, sum_dists);
}

#endif //METRICS_HPP
