#include "calc.h"

/* set inlets velocity there are two type inlets*/
int set_inlets(const t_param params, float *inlets) {
#if defined(LBM_ENV_AUTOLAB)
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(inlets) num_threads(8)
#else
#pragma omp parallel for default(none) shared(params, inlets) num_threads(8)
#endif
#else
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(inlets) \
    num_threads(omp_get_num_procs())
#else
#pragma omp parallel for default(none) shared(params, inlets) \
    num_threads(omp_get_num_procs())
#endif
#endif
  for (int jj = 0; jj < params.ny; jj++) {
    if (!params.type)
      inlets[jj] = params.velocity;  // homogeneous
    else
      inlets[jj] = params.velocity * 4.0 *
                   ((1 - ((float)jj) / params.ny) * ((float)(jj + 1)) /
                    params.ny);  // parabolic
  }
  return EXIT_SUCCESS;
}

/* compute average velocity of whole grid, ignore grids with obstacles. */
float av_velocity(const t_param params, t_speed *cells, int *obstacles) {
  int tot_cells = 0; /* no. of cells used in calculation */
  float tot_u;       /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
#if defined(LBM_ENV_AUTOLAB)
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, obstacles) \
    reduction(+ : tot_cells) reduction(+ : tot_u) num_threads(8)
#else
#pragma omp parallel for default(none) shared(params, cells, obstacles) \
    reduction(+ : tot_cells) reduction(+ : tot_u) num_threads(8)
#endif
#else
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, obstacles) \
    reduction(+ : tot_cells) reduction(+ : tot_u)               \
    num_threads(omp_get_num_procs())
#else
#pragma omp parallel for default(none) shared(params, cells, obstacles) \
    reduction(+ : tot_cells) reduction(+ : tot_u)                       \
    num_threads(omp_get_num_procs())
#endif
#endif
  for (int jj = 0; jj < params.ny; jj++) {
    float local_density; /* per grid cell sum of densities */
    float u_x;           /* x-component of velocity in grid cell */
    float u_y;           /* y-component of velocity in grid cell */

    float local_density_1_2 = 0.f;
    float local_density_3_4 = 0.f;
    float local_density_5_6 = 0.f;
    float local_density_7_8 = 0.f;

    int index = 0;
    for (int ii = 0; ii < params.nx; ii++) {
      /* ignore occupied cells */
      index = ii + jj * params.nx;
      if (!obstacles[index]) {
        /* local density total */

        local_density = cells->speeds_0[index];

        local_density_1_2 = cells->speeds_1_8[index].speeds_1_8[0] +
                            cells->speeds_1_8[index].speeds_1_8[1];
        local_density_3_4 = cells->speeds_1_8[index].speeds_1_8[2] +
                            cells->speeds_1_8[index].speeds_1_8[3];
        local_density_5_6 = cells->speeds_1_8[index].speeds_1_8[4] +
                            cells->speeds_1_8[index].speeds_1_8[5];
        local_density_7_8 = cells->speeds_1_8[index].speeds_1_8[6] +
                            cells->speeds_1_8[index].speeds_1_8[7];

        local_density += local_density_1_2 + local_density_3_4 +
                         local_density_5_6 + local_density_7_8;

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

        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        tot_cells += 1;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

/* calculate reynold number */
float calc_reynolds(const t_param params, t_speed *cells, int *obstacles) {
  return av_velocity(params, cells, obstacles) * (float)(params.ny) /
         params.viscosity;
}