#ifndef UTILS_H
#define UTILS_H

#include "types.h"

/* This file contains utility functions for initialise, finalise, and outputting the corresponding files */
void die(const char* message, const int line, const char* file);

/* Load parameters, allocate memory, initialize grids and obstacle. */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** inlets_ptr);

/* Finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** inlets);

/* Output current grid state -- each cells' velocity. */
int write_state(char* filename, const t_param params, t_speed* cells, int* obstacles);

//rewrite initialise in algined_t_speed
int initialise_aligned(const char* paramfile, const char* obstaclefile,
               t_param* params, aligned_t_speed*cells_ptr, aligned_t_speed*tmp_cells_ptr,
               int** obstacles_ptr, float** inlets_ptr);
// rewrite write_state in algined_t_speed
int write_state_aligned(char* filename, const t_param params, aligned_t_speed cells, int* obstacles);
// rewrite finalise in algined_t_speed
int finalise_aligned(const t_param* params, aligned_t_speed cells_ptr, aligned_t_speed tmp_cells_ptr,
             int** obstacles_ptr, float** inlets);


void* aligned_malloc(size_t required_bytes, size_t alignment);
void aligned_free(void *p);

#endif