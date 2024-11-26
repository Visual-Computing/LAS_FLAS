//
// Created by Bruno Schilling on 10/31/24.
//

#ifndef DET_RANDOM_H
#define DET_RANDOM_H

typedef struct {
  unsigned int seed;
} RandomGenerator;

RandomGenerator *get_instance(unsigned int seed);
unsigned int det_rand();
unsigned int det_next_int(unsigned int bound);
float det_next_float();

#endif //DET_RANDOM_H
