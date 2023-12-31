#include "d2q9_bgk.h"

/* The main processes in one step */
int collision(const t_param params, t_speed *cells, t_speed *tmp_cells,
              bool *obstacles);
int streaming(const t_param params, t_speed *cells, t_speed *tmp_cells);
int obstacle(const t_param params, t_speed *cells, t_speed *tmp_cells,
             bool *obstacles);
int boundary(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** collision(), obstacle(), streaming() & boundary()
*/
int timestep(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets, bool *obstacles) {
  /* The main time overhead, you should mainly optimize these processes. */
  collision(params, cells, tmp_cells, obstacles);
  // obstacle(params, cells, tmp_cells, obstacles);
  streaming(params, cells, tmp_cells);
  boundary(params, cells, tmp_cells, inlets);
  return EXIT_SUCCESS;
}

/*
** The collision of fluids in the cell is calculated using
** the local equilibrium distribution and relaxation process
*/
int collision(const t_param params, t_speed *cells, t_speed *tmp_cells,
              bool *obstacles) {
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */

  const __m256 one_vec = _mm256_set1_ps(1.f);
  const __m256 half_vec = _mm256_set1_ps(0.5f);
  const __m256 w_vec = _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2);
  const __m256 one_div_c_sq_vec = _mm256_set1_ps(1.f / c_sq);
  const __m256 omega_vec = _mm256_set1_ps(params.omega);

  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */
#if defined(LBM_ENV_AUTOLAB)
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, tmp_cells, obstacles) \
    num_threads(8)
#else
#pragma omp parallel for default(none)                                     \
    shared(params, cells, tmp_cells, obstacles, c_sq, w0, w1, w2, one_vec, \
               half_vec, w_vec, one_div_c_sq_vec, omega_vec) num_threads(8)
#endif
#else
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, tmp_cells, obstacles) \
    num_threads(omp_get_num_procs())
#else
#pragma omp parallel for default(none)                                     \
    shared(params, cells, tmp_cells, obstacles, c_sq, w0, w1, w2, one_vec, \
               half_vec, w_vec, one_div_c_sq_vec, omega_vec)               \
    num_threads(omp_get_num_procs())
#endif
#endif
  for (int jj = 0; jj < params.ny; jj++) {
    float local_density = 0.f;
    float u_x = 0.f;
    float u_y = 0.f;
    float u_sq = 0.f;
    float local_density_sq = 0.f;

    float local_density_1_2 = 0.f;
    float local_density_3_4 = 0.f;
    float local_density_5_6 = 0.f;
    float local_density_7_8 = 0.f;

    int index = 0;
    int y_n = 0;
    int y_s = 0;
    int x_e = 0;
    int x_w = 0;
    t_speed_0 tmp_cell_0 = 0.f;
    t_speed_1_8 tmp_cell_1_8 = {};
    for (int ii = 0; ii < params.nx; ii++) {
      index = ii + jj * params.nx;
      x_e = (ii + 1) & (params.nx - 1);
      x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);
      if (obstacles[index]) {
        /* called after collision, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cell_0 = cells->speeds_0[index];

        tmp_cell_1_8.speeds_1_8[0] = cells->speeds_1_8[index].speeds_1_8[2];
        tmp_cell_1_8.speeds_1_8[1] = cells->speeds_1_8[index].speeds_1_8[3];
        tmp_cell_1_8.speeds_1_8[2] = cells->speeds_1_8[index].speeds_1_8[0];
        tmp_cell_1_8.speeds_1_8[3] = cells->speeds_1_8[index].speeds_1_8[1];
        tmp_cell_1_8.speeds_1_8[4] = cells->speeds_1_8[index].speeds_1_8[6];
        tmp_cell_1_8.speeds_1_8[5] = cells->speeds_1_8[index].speeds_1_8[7];
        tmp_cell_1_8.speeds_1_8[6] = cells->speeds_1_8[index].speeds_1_8[4];
        tmp_cell_1_8.speeds_1_8[7] = cells->speeds_1_8[index].speeds_1_8[5];
      } else {
        /* compute local density total */
        local_density = cells->speeds_0[index];

        local_density_1_2 = cells->speeds_1_8[index].speeds_1_8[0] +
                            cells->speeds_1_8[index].speeds_1_8[1];
        local_density_3_4 = cells->speeds_1_8[index].speeds_1_8[2] +
                            cells->speeds_1_8[index].speeds_1_8[3];
        local_density_5_6 = cells->speeds_1_8[index].speeds_1_8[4] +
                            cells->speeds_1_8[index].speeds_1_8[5];
        local_density_7_8 = cells->speeds_1_8[index].speeds_1_8[6] +
                            cells->speeds_1_8[index].speeds_1_8[7];

        local_density_1_2 += local_density_3_4;
        local_density_5_6 += local_density_7_8;

        local_density += local_density_1_2 + local_density_5_6;

        __m256 cells_vec = _mm256_loadu_ps(cells->speeds_1_8[index].speeds_1_8);

        /* compute x velocity component */
        u_x = cells->speeds_1_8[index].speeds_1_8[0];
        u_y = cells->speeds_1_8[index].speeds_1_8[1];

        u_x -= cells->speeds_1_8[index].speeds_1_8[2];
        u_y -= cells->speeds_1_8[index].speeds_1_8[3];

        u_x += cells->speeds_1_8[index].speeds_1_8[4];
        u_y += cells->speeds_1_8[index].speeds_1_8[4];

        u_x -= cells->speeds_1_8[index].speeds_1_8[5];
        u_y += cells->speeds_1_8[index].speeds_1_8[5];

        u_x -= cells->speeds_1_8[index].speeds_1_8[6];
        u_y -= cells->speeds_1_8[index].speeds_1_8[6];

        u_x += cells->speeds_1_8[index].speeds_1_8[7];
        u_y -= cells->speeds_1_8[index].speeds_1_8[7];

        /* divide by rho */
        u_x /= local_density;
        u_y /= local_density;

        /* velocity squared */
        u_sq = u_x * u_x + u_y * u_y;

        __m256 u_vec = _mm256_setr_ps(u_x, u_y, -u_x, -u_y, u_x + u_y,
                                      -u_x + u_y, -u_x - u_y, u_x - u_y);

        local_density_sq = local_density * (1.f - u_sq / (2.f * c_sq));

        /* directional velocity components */
        __m256 local_density_vec = _mm256_set1_ps(local_density);
        __m256 local_density_sq_vec = _mm256_set1_ps(local_density_sq);

        /* equilibrium densities */
        __m256 u_sq_vec = _mm256_mul_ps(u_vec, one_div_c_sq_vec);

        __m256 temp_vec = _mm256_mul_ps(half_vec, u_sq_vec);
        temp_vec = _mm256_add_ps(one_vec, temp_vec);
        temp_vec = _mm256_mul_ps(u_sq_vec, temp_vec);
        temp_vec = _mm256_mul_ps(local_density_vec, temp_vec);
        temp_vec = _mm256_add_ps(local_density_sq_vec, temp_vec);
        __m256 d_equ_vec = _mm256_mul_ps(w_vec, temp_vec);

        temp_vec = _mm256_sub_ps(d_equ_vec, cells_vec);
        temp_vec = _mm256_mul_ps(omega_vec, temp_vec);
        temp_vec = _mm256_add_ps(cells_vec, temp_vec);
        _mm256_storeu_ps(tmp_cell_1_8.speeds_1_8, temp_vec);

        tmp_cell_0 =
            cells->speeds_0[index] +
            params.omega * (w0 * local_density_sq - cells->speeds_0[index]);
      }

      tmp_cells->speeds_0[index] = tmp_cell_0;

      tmp_cells->speeds_1_8[x_e + jj * params.nx].speeds_1_8[0] =
          tmp_cell_1_8.speeds_1_8[0]; /* east */
      tmp_cells->speeds_1_8[x_w + jj * params.nx].speeds_1_8[2] =
          tmp_cell_1_8.speeds_1_8[2]; /* west */

      if (jj == 0) {
        y_s = 1;
        tmp_cells->speeds_1_8[index].speeds_1_8[7] =
            tmp_cell_1_8.speeds_1_8[5]; /* north-west */
        tmp_cells->speeds_1_8[index].speeds_1_8[3] =
            tmp_cell_1_8.speeds_1_8[1]; /* north */
        tmp_cells->speeds_1_8[index].speeds_1_8[6] =
            tmp_cell_1_8.speeds_1_8[4]; /* north-east */

        tmp_cells->speeds_1_8[x_w + y_s * params.nx].speeds_1_8[6] =
            tmp_cell_1_8.speeds_1_8[6]; /* south-west */
        tmp_cells->speeds_1_8[ii + y_s * params.nx].speeds_1_8[3] =
            tmp_cell_1_8.speeds_1_8[3]; /* south */
        tmp_cells->speeds_1_8[x_e + y_s * params.nx].speeds_1_8[7] =
            tmp_cell_1_8.speeds_1_8[7]; /* south-east */
      } else if (jj == params.ny - 1) {
        y_n = params.ny - 2;
        tmp_cells->speeds_1_8[x_w + y_n * params.nx].speeds_1_8[5] =
            tmp_cell_1_8.speeds_1_8[5]; /* north-west */
        tmp_cells->speeds_1_8[ii + y_n * params.nx].speeds_1_8[1] =
            tmp_cell_1_8.speeds_1_8[1]; /* north */
        tmp_cells->speeds_1_8[x_e + y_n * params.nx].speeds_1_8[4] =
            tmp_cell_1_8.speeds_1_8[4]; /* north-east */

        tmp_cells->speeds_1_8[index].speeds_1_8[4] =
            tmp_cell_1_8.speeds_1_8[6]; /* south-west */
        tmp_cells->speeds_1_8[index].speeds_1_8[1] =
            tmp_cell_1_8.speeds_1_8[3]; /* south */
        tmp_cells->speeds_1_8[index].speeds_1_8[5] =
            tmp_cell_1_8.speeds_1_8[7]; /* south-east */
      } else {
        y_s = jj + 1;
        y_n = jj - 1;
        tmp_cells->speeds_1_8[x_w + y_n * params.nx].speeds_1_8[5] =
            tmp_cell_1_8.speeds_1_8[5]; /* north-west */
        tmp_cells->speeds_1_8[ii + y_n * params.nx].speeds_1_8[1] =
            tmp_cell_1_8.speeds_1_8[1]; /* north */
        tmp_cells->speeds_1_8[x_e + y_n * params.nx].speeds_1_8[4] =
            tmp_cell_1_8.speeds_1_8[4]; /* north-east */

        tmp_cells->speeds_1_8[x_w + y_s * params.nx].speeds_1_8[6] =
            tmp_cell_1_8.speeds_1_8[6]; /* south-west */
        tmp_cells->speeds_1_8[ii + y_s * params.nx].speeds_1_8[3] =
            tmp_cell_1_8.speeds_1_8[3]; /* south */
        tmp_cells->speeds_1_8[x_e + y_s * params.nx].speeds_1_8[7] =
            tmp_cell_1_8.speeds_1_8[7]; /* south-east */
      }
    }
  }

  return EXIT_SUCCESS;
}

/*
** For obstacles, mirror their speed.
*/
int obstacle(const t_param params, t_speed *cells, t_speed *tmp_cells,
             bool *obstacles) {
/* loop over the cells in the grid */
#if defined(LBM_ENV_AUTOLAB)
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, tmp_cells, obstacles) \
    num_threads(8)
#else
#pragma omp parallel for default(none) \
    shared(params, cells, tmp_cells, obstacles) num_threads(8)
#endif
#else
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, tmp_cells, obstacles) \
    num_threads(omp_get_num_procs())
#else
#pragma omp parallel for default(none) shared( \
        params, cells, tmp_cells, obstacles) num_threads(omp_get_num_procs())
#endif
#endif
  for (int jj = 0; jj < params.ny; jj++) {
    int index = 0;
    for (int ii = 0; ii < params.nx; ii++) {
      /* if the cell contains an obstacle */
      index = ii + jj * params.nx;
      if (obstacles[index]) {
        /* called after collision, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells->speeds_0[index] = cells->speeds_0[index];

        tmp_cells->speeds_1_8[index].speeds_1_8[0] =
            cells->speeds_1_8[index].speeds_1_8[2];
        tmp_cells->speeds_1_8[index].speeds_1_8[1] =
            cells->speeds_1_8[index].speeds_1_8[3];
        tmp_cells->speeds_1_8[index].speeds_1_8[2] =
            cells->speeds_1_8[index].speeds_1_8[0];
        tmp_cells->speeds_1_8[index].speeds_1_8[3] =
            cells->speeds_1_8[index].speeds_1_8[1];
        tmp_cells->speeds_1_8[index].speeds_1_8[4] =
            cells->speeds_1_8[index].speeds_1_8[6];
        tmp_cells->speeds_1_8[index].speeds_1_8[5] =
            cells->speeds_1_8[index].speeds_1_8[7];
        tmp_cells->speeds_1_8[index].speeds_1_8[6] =
            cells->speeds_1_8[index].speeds_1_8[4];
        tmp_cells->speeds_1_8[index].speeds_1_8[7] =
            cells->speeds_1_8[index].speeds_1_8[5];
      }
    }
  }
  return EXIT_SUCCESS;
}

static inline void swap_ptr(void **a, void **b) {
  void *tmp = *a;
  *a = *b;
  *b = tmp;
}

/*
** Particles flow to the corresponding cell according to their speed direaction.
*/
int streaming(const t_param params, t_speed *cells, t_speed *tmp_cells) {
  swap_ptr((void **)&cells->speeds_0, (void **)&tmp_cells->speeds_0);
  swap_ptr((void **)&cells->speeds_1_8, (void **)&tmp_cells->speeds_1_8);
  return EXIT_SUCCESS;
}

/*
** Work with boundary conditions. The upper and lower boundaries use the rebound
*plane,
** the left border is the inlet of fixed speed, and
** the right border is the open outlet of the first-order approximation.
*/
int boundary(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets) {
  /* Set the constant coefficient */
  const float cst1 = 2.0 / 3.0;
  const float cst2 = 1.0 / 6.0;
  const float cst3 = 1.0 / 2.0;

#if defined(LBM_ENV_AUTOLAB)
#if __GNUC__ < 9
#pragma omp parallel default(none) shared(cells, tmp_cells, inlets) \
    num_threads(8)
#else
#pragma omp parallel default(none) \
    shared(params, cells, tmp_cells, inlets, cst1, cst2, cst3) num_threads(8)
#endif
#else
#if __GNUC__ < 9
#pragma omp parallel default(none) shared(cells, tmp_cells, inlets) \
    num_threads(omp_get_num_procs())
#else
#pragma omp parallel default(none)                             \
    shared(params, cells, tmp_cells, inlets, cst1, cst2, cst3) \
    num_threads(omp_get_num_procs())
#endif
#endif
  {
    // left wall (inlet)
#pragma omp for nowait
    for (int jj = 0; jj < params.ny; jj++) {
      int ii = 0;
      int index = ii + jj * params.nx;
      float local_density =
          (cells->speeds_0[index] + cells->speeds_1_8[index].speeds_1_8[1] +
           cells->speeds_1_8[index].speeds_1_8[3] +
           2.f * cells->speeds_1_8[index].speeds_1_8[2] +
           2.f * cells->speeds_1_8[index].speeds_1_8[5] +
           2.f * cells->speeds_1_8[index].speeds_1_8[6]) /
          (1.f - inlets[jj]);

      cells->speeds_1_8[index].speeds_1_8[0] =
          cells->speeds_1_8[index].speeds_1_8[2] +
          cst1 * local_density * inlets[jj];

      cells->speeds_1_8[index].speeds_1_8[4] =
          cells->speeds_1_8[index].speeds_1_8[6] -
          cst3 * (cells->speeds_1_8[index].speeds_1_8[1] -
                  cells->speeds_1_8[index].speeds_1_8[3]) +
          cst2 * local_density * inlets[jj];

      cells->speeds_1_8[index].speeds_1_8[7] =
          cells->speeds_1_8[index].speeds_1_8[5] +
          cst3 * (cells->speeds_1_8[index].speeds_1_8[1] -
                  cells->speeds_1_8[index].speeds_1_8[3]) +
          cst2 * local_density * inlets[jj];
    }

    // right wall (outlet)
#pragma omp for nowait
    for (int jj = 0; jj < params.ny; jj++) {
      int ii = params.nx - 1;
      int index = ii + jj * params.nx;
      cells->speeds_0[index] = cells->speeds_0[ii - 1 + jj * params.nx];
      for (int kk = 0; kk < 8; kk++) {
        cells->speeds_1_8[index].speeds_1_8[kk] =
            cells->speeds_1_8[ii - 1 + jj * params.nx].speeds_1_8[kk];
      }
    }

    // wait for all threads to complete
#pragma omp barrier
  }

  return EXIT_SUCCESS;
}
