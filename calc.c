#include "calc.h"

/* set inlets velocity there are two type inlets*/
int set_inlets(const t_param params, float *inlets) {
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
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* ignore occupied cells */
      if (!obstacles[ii + jj * params.nx]) {
        /* local density total */
        float local_density = cells->speeds_0[ii + jj * params.nx];

        for (int kk = 0; kk < 8; kk++) {
          local_density +=
              cells->speeds_1_8[ii + jj * params.nx].speeds_1_8[kk];
        }

        /* x-component of velocity */
        float u_x = (cells->speeds_1_8[ii + jj * params.nx].speeds_1_8[0] +
                     cells->speeds_1_8[ii + jj * params.nx].speeds_1_8[4] +
                     cells->speeds_1_8[ii + jj * params.nx].speeds_1_8[7] -
                     (cells->speeds_1_8[ii + jj * params.nx].speeds_1_8[2] +
                      cells->speeds_1_8[ii + jj * params.nx].speeds_1_8[5] +
                      cells->speeds_1_8[ii + jj * params.nx].speeds_1_8[6])) /
                    local_density;
        /* compute y velocity component */
        float u_y = (cells->speeds_1_8[ii + jj * params.nx].speeds_1_8[1] +
                     cells->speeds_1_8[ii + jj * params.nx].speeds_1_8[4] +
                     cells->speeds_1_8[ii + jj * params.nx].speeds_1_8[5] -
                     (cells->speeds_1_8[ii + jj * params.nx].speeds_1_8[3] +
                      cells->speeds_1_8[ii + jj * params.nx].speeds_1_8[6] +
                      cells->speeds_1_8[ii + jj * params.nx].speeds_1_8[7])) /
                    local_density;

        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
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