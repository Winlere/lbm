#ifndef TYPES_H
#define TYPES_H

#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <tmmintrin.h>

#define NSPEEDS 9
#define NUM_THREADS 28

typedef struct {
  int nx;          /* no. of cells in x-direction */
  int ny;          /* no. of cells in y-direction */
  int maxIters;    /* no. of iterations */
  float density;   /* density per cell */
  float viscosity; /* kinematic viscosity of fluid */
  float velocity;  /* inlet velocity */
  int type;        /* inlet type */
  float omega;     /* relaxation parameter */
} t_param;

/* struct to hold the distribution of different speeds */
typedef struct {
  float speeds_1_8[8];
} __attribute__((aligned(32))) t_speed_1_8;

typedef float t_speed_0;

typedef struct {
  t_speed_1_8* speeds_1_8;
  t_speed_0* speeds_0;
} t_speed;

#endif