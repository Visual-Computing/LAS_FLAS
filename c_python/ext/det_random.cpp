//
// Created by Bruno Schilling on 10/31/24.
//

#include "det_random.hpp"

RandomGenerator *get_instance(unsigned int seed) {
  static RandomGenerator instance;
  static int initialized = 0;

  if (!initialized) {
    instance.seed = seed;
    initialized = 1;
  }

  return &instance;
}

unsigned int det_rand() {
  RandomGenerator *generator = get_instance(123);
  generator->seed = (generator->seed * 1103515245 + 12345) & 0x7fffffff;
  return generator->seed;
}

unsigned int det_next_int(unsigned int bound) {
  unsigned int n = det_rand();
  return n % bound;
}

float det_next_float() {
  unsigned int n = det_rand();
  return static_cast<float>(n & 0x7fffffff) / static_cast<float>(0x7fffffff);
}
