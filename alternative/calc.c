#include "calc.h"

/* set inlets velocity there are two type inlets*/
int set_inlets(const t_param params, float* inlets) {
  for(int jj=0; jj <params.ny; jj++){
    if(!params.type)
      inlets[jj]=params.velocity; // homogeneous
    else
      inlets[jj]=params.velocity * 4.0 *((1-((float)jj)/params.ny)*((float)(jj+1))/params.ny); // parabolic
  }
  return EXIT_SUCCESS;
}

/* compute average velocity of whole grid, ignore grids with obstacles. */
float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float  tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii + jj*params.nx].speeds[1]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[8]
                      - (cells[ii + jj*params.nx].speeds[3]
                         + cells[ii + jj*params.nx].speeds[6]
                         + cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[6]
                      - (cells[ii + jj*params.nx].speeds[4]
                         + cells[ii + jj*params.nx].speeds[7]
                         + cells[ii + jj*params.nx].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

float aligned_av_velocity(const t_param params, aligned_t_speed cells, int *obstacles){
    int    tot_cells = 0;  /* no. of cells used in calculation */
  float  tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = cells.stay[ii + jj*params.nx];

        for (int kk = 0; kk < NSPEEDS - 1; kk++)
        {
          local_density += cells.other[ii + jj*params.nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells.other[ii + jj*params.nx].speeds[1 - 1]
                      + cells.other[ii + jj*params.nx].speeds[5 - 1]
                      + cells.other[ii + jj*params.nx].speeds[8 - 1]
                      - (cells.other[ii + jj*params.nx].speeds[3 - 1]
                         + cells.other[ii + jj*params.nx].speeds[6 - 1]
                         + cells.other[ii + jj*params.nx].speeds[7 - 1]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells.other[ii + jj*params.nx].speeds[2 - 1]
                      + cells.other[ii + jj*params.nx].speeds[5 - 1]
                      + cells.other[ii + jj*params.nx].speeds[6 - 1]
                      - (cells.other[ii + jj*params.nx].speeds[4 - 1]
                         + cells.other[ii + jj*params.nx].speeds[7 - 1]
                         + cells.other[ii + jj*params.nx].speeds[8 - 1]) )
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

float aligned_calc_reynolds(const t_param params, aligned_t_speed cells, int *obstacles)
{
  return aligned_av_velocity(params, cells, obstacles) * (float)(params.ny) / params.viscosity;
}

/* calculate reynold number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  return av_velocity(params, cells, obstacles) * (float)(params.ny) / params.viscosity;
}