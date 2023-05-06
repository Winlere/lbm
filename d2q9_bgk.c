#include "d2q9_bgk.h"

/* The main processes in one step */
int collision(const t_param params, t_speed *cells, t_speed *tmp_cells,
              int *obstacles);
int streaming(const t_param params, t_speed *cells, t_speed *tmp_cells);
int obstacle(const t_param params, t_speed *cells, t_speed *tmp_cells,
             int *obstacles);
int boundary(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** collision(), obstacle(), streaming() & boundary()
*/
int timestep(const t_param params, t_speed *cells, t_speed *tmp_cells,
             float *inlets, int *obstacles) {
  /* The main time overhead, you should mainly optimize these processes. */
  collision(params, cells, tmp_cells, obstacles);
  obstacle(params, cells, tmp_cells, obstacles);
  streaming(params, cells, tmp_cells);
  boundary(params, cells, tmp_cells, inlets);
  return EXIT_SUCCESS;
}

/*
** The collision of fluids in the cell is calculated using
** the local equilibrium distribution and relaxation process
*/
int collision(const t_param params, t_speed *cells, t_speed *tmp_cells,
              int *obstacles) {
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */

  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */
#if defined(LBM_ENV_AUTOLAB)
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, tmp_cells, obstacles) \
    num_threads(4)
#else
#pragma omp parallel for default(none) shared( \
        params, cells, tmp_cells, obstacles, c_sq, w0, w1, w2) num_threads(4)
#endif
#else
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, tmp_cells, obstacles)
#else
#pragma omp parallel for default(none) \
    shared(params, cells, tmp_cells, obstacles, c_sq, w0, w1, w2)
#endif
#endif
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      if (!obstacles[ii + jj * params.nx]) {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++) {
          local_density += cells[ii + jj * params.nx].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (cells[ii + jj * params.nx].speeds[1] +
                     cells[ii + jj * params.nx].speeds[5] +
                     cells[ii + jj * params.nx].speeds[8] -
                     (cells[ii + jj * params.nx].speeds[3] +
                      cells[ii + jj * params.nx].speeds[6] +
                      cells[ii + jj * params.nx].speeds[7])) /
                    local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj * params.nx].speeds[2] +
                     cells[ii + jj * params.nx].speeds[5] +
                     cells[ii + jj * params.nx].speeds[6] -
                     (cells[ii + jj * params.nx].speeds[4] +
                      cells[ii + jj * params.nx].speeds[7] +
                      cells[ii + jj * params.nx].speeds[8])) /
                    local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        float local_density_sq = local_density * (1.f - u_sq / (2.f * c_sq));

        tmp_cells[ii + jj * params.nx].speeds[0] =
            cells[ii + jj * params.nx].speeds[0] +
            params.omega *
                (w0 * local_density_sq - cells[ii + jj * params.nx].speeds[0]);

        /* directional velocity components */
        __m256 u_sq_vec = _mm256_div_ps(
            _mm256_add_ps(
                _mm256_setr_ps(u_x, u_y, -u_x, -u_y, u_x, -u_x, -u_x, u_x),
                _mm256_setr_ps(0, 0, 0, 0, u_y, u_y, -u_y, -u_y)),
            _mm256_set1_ps(c_sq));

        /* equilibrium densities */
        __m256 w_vec = _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2);
        __m256 cells_vec =
            _mm256_loadu_ps(cells[ii + jj * params.nx].speeds + 1);
        __m256 d_equ_vec = _mm256_mul_ps(
            w_vec, _mm256_add_ps(
                       _mm256_set1_ps(local_density_sq),
                       _mm256_mul_ps(
                           _mm256_set1_ps(local_density),
                           _mm256_mul_ps(
                               u_sq_vec,
                               _mm256_add_ps(_mm256_set1_ps(1.f),
                                             _mm256_mul_ps(_mm256_set1_ps(.5f),
                                                           u_sq_vec))))));

        _mm256_storeu_ps(
            tmp_cells[ii + jj * params.nx].speeds + 1,
            _mm256_add_ps(cells_vec,
                          _mm256_mul_ps(_mm256_set1_ps(params.omega),
                                        _mm256_sub_ps(d_equ_vec, cells_vec))));
      }
    }
  }
  return EXIT_SUCCESS;
}

/*
** For obstacles, mirror their speed.
*/
int obstacle(const t_param params, t_speed *cells, t_speed *tmp_cells,
             int *obstacles) {
/* loop over the cells in the grid */
#if defined(LBM_ENV_AUTOLAB)
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, tmp_cells, obstacles) \
    num_threads(4)
#else
#pragma omp parallel for default(none) \
    shared(params, cells, tmp_cells, obstacles) num_threads(4)
#endif
#else
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, tmp_cells, obstacles)
#else
#pragma omp parallel for default(none) \
    shared(params, cells, tmp_cells, obstacles)
#endif
#endif
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* if the cell contains an obstacle */
      if (obstacles[jj * params.nx + ii]) {
        /* called after collision, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells[ii + jj * params.nx].speeds[0] =
            cells[ii + jj * params.nx].speeds[0];
        tmp_cells[ii + jj * params.nx].speeds[1] =
            cells[ii + jj * params.nx].speeds[3];
        tmp_cells[ii + jj * params.nx].speeds[2] =
            cells[ii + jj * params.nx].speeds[4];
        tmp_cells[ii + jj * params.nx].speeds[3] =
            cells[ii + jj * params.nx].speeds[1];
        tmp_cells[ii + jj * params.nx].speeds[4] =
            cells[ii + jj * params.nx].speeds[2];
        tmp_cells[ii + jj * params.nx].speeds[5] =
            cells[ii + jj * params.nx].speeds[7];
        tmp_cells[ii + jj * params.nx].speeds[6] =
            cells[ii + jj * params.nx].speeds[8];
        tmp_cells[ii + jj * params.nx].speeds[7] =
            cells[ii + jj * params.nx].speeds[5];
        tmp_cells[ii + jj * params.nx].speeds[8] =
            cells[ii + jj * params.nx].speeds[6];
      }
    }
  }
  return EXIT_SUCCESS;
}

/*
** Particles flow to the corresponding cell according to their speed direaction.
*/
int streaming(const t_param params, t_speed *cells, t_speed *tmp_cells) {
/* loop over _all_ cells */
#if defined(LBM_ENV_AUTOLAB)
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, tmp_cells) num_threads(4)
#else
#pragma omp parallel for default(none) shared(params, cells, tmp_cells) \
    num_threads(4)
#endif
#else
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, tmp_cells)
#else
#pragma omp parallel for default(none) shared(params, cells, tmp_cells)
#endif
#endif
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      cells[ii + jj * params.nx].speeds[0] =
          tmp_cells[ii + jj * params.nx]
              .speeds[0]; /* central cell, no movement */
      cells[ii + jj * params.nx].speeds[1] =
          tmp_cells[x_w + jj * params.nx].speeds[1]; /* east */
      cells[ii + jj * params.nx].speeds[3] =
          tmp_cells[x_e + jj * params.nx].speeds[3]; /* west */

      cells[ii + jj * params.nx].speeds[2] =
          tmp_cells[ii + y_s * params.nx].speeds[2]; /* south */
      cells[ii + jj * params.nx].speeds[5] =
          tmp_cells[x_w + y_s * params.nx].speeds[5]; /* north-east */
      cells[ii + jj * params.nx].speeds[6] =
          tmp_cells[x_e + y_s * params.nx].speeds[6]; /* north-west */

      cells[ii + jj * params.nx].speeds[4] =
          tmp_cells[ii + y_n * params.nx].speeds[4]; /* south */
      cells[ii + jj * params.nx].speeds[8] =
          tmp_cells[x_w + y_n * params.nx].speeds[8]; /* south-east */
      cells[ii + jj * params.nx].speeds[7] =
          tmp_cells[x_e + y_n * params.nx].speeds[7]; /* south-west */
    }
  }

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
    num_threads(4)
#else
#pragma omp parallel default(none) \
    shared(params, cells, tmp_cells, inlets, cst1, cst2, cst3) num_threads(4)
#endif
#else
#if __GNUC__ < 9
#pragma omp parallel default(none) shared(cells, tmp_cells, inlets)
#else
#pragma omp parallel default(none) \
    shared(params, cells, tmp_cells, inlets, cst1, cst2, cst3)
#endif
#endif
  {
    // top wall (bounce)
#pragma omp for nowait
    for (int ii = 0; ii < params.nx; ii++) {
      int jj = params.ny - 1;
      cells[ii + jj * params.nx].speeds[4] =
          tmp_cells[ii + jj * params.nx].speeds[2];
      cells[ii + jj * params.nx].speeds[7] =
          tmp_cells[ii + jj * params.nx].speeds[5];
      cells[ii + jj * params.nx].speeds[8] =
          tmp_cells[ii + jj * params.nx].speeds[6];
    }

    // bottom wall (bounce)
#pragma omp for nowait
    for (int ii = 0; ii < params.nx; ii++) {
      int jj = 0;
      cells[ii + jj * params.nx].speeds[2] =
          tmp_cells[ii + jj * params.nx].speeds[4];
      cells[ii + jj * params.nx].speeds[5] =
          tmp_cells[ii + jj * params.nx].speeds[7];
      cells[ii + jj * params.nx].speeds[6] =
          tmp_cells[ii + jj * params.nx].speeds[8];
    }
    // wait for all threads to complete
#pragma omp barrier
  }

#if defined(LBM_ENV_AUTOLAB)
#if __GNUC__ < 9
#pragma omp parallel default(none) shared(cells, tmp_cells, inlets) \
    num_threads(4)
#else
#pragma omp parallel default(none) \
    shared(params, cells, tmp_cells, inlets, cst1, cst2, cst3) num_threads(4)
#endif
#else
#if __GNUC__ < 9
#pragma omp parallel default(none) shared(cells, tmp_cells, inlets)
#else
#pragma omp parallel default(none) \
    shared(params, cells, tmp_cells, inlets, cst1, cst2, cst3)
#endif
#endif
  {
    // left wall (inlet)
#pragma omp for nowait
    for (int jj = 0; jj < params.ny; jj++) {
      int ii = 0;
      float local_density = (cells[ii + jj * params.nx].speeds[0] +
                             cells[ii + jj * params.nx].speeds[2] +
                             cells[ii + jj * params.nx].speeds[4] +
                             2.0 * cells[ii + jj * params.nx].speeds[3] +
                             2.0 * cells[ii + jj * params.nx].speeds[6] +
                             2.0 * cells[ii + jj * params.nx].speeds[7]) /
                            (1.0 - inlets[jj]);

      cells[ii + jj * params.nx].speeds[1] =
          cells[ii + jj * params.nx].speeds[3] +
          cst1 * local_density * inlets[jj];

      cells[ii + jj * params.nx].speeds[5] =
          cells[ii + jj * params.nx].speeds[7] -
          cst3 * (cells[ii + jj * params.nx].speeds[2] -
                  cells[ii + jj * params.nx].speeds[4]) +
          cst2 * local_density * inlets[jj];

      cells[ii + jj * params.nx].speeds[8] =
          cells[ii + jj * params.nx].speeds[6] +
          cst3 * (cells[ii + jj * params.nx].speeds[2] -
                  cells[ii + jj * params.nx].speeds[4]) +
          cst2 * local_density * inlets[jj];
    }

    // right wall (outlet)
#pragma omp for nowait
    for (int jj = 0; jj < params.ny; jj++) {
      int ii = params.nx - 1;
      for (int kk = 0; kk < NSPEEDS; kk++) {
        cells[ii + jj * params.nx].speeds[kk] =
            cells[ii - 1 + jj * params.nx].speeds[kk];
      }
    }

    // wait for all threads to complete
#pragma omp barrier
  }

  return EXIT_SUCCESS;
}
