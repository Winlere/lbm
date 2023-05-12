#include "types.h"

/* utility functions */
void die(const char *message, const int line, const char *file) {
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

/* load params, allocate memory, load obstacles & initialise fluid particle
 * densities */
int initialise(const char *paramfile, const char *obstaclefile, t_param *params,
               t_speed *cells_ptr, t_speed *tmp_cells_ptr, int **obstacles_ptr,
               float **inlets_ptr) {
  char message[1024]; /* message buffer */
  FILE *fp;           /* file pointer */
  int xx, yy;         /* generic array indices */
  int blocked;        /* indicates whether a cell is blocked by an obstacle */
  int retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter nx */
  retval = fscanf(fp, "nx: %d\n", &(params->nx));
  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  /* read in the parameter ny */
  retval = fscanf(fp, "ny: %d\n", &(params->ny));
  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  /* read in the parameter maxIters */
  retval = fscanf(fp, "iters: %d\n", &(params->maxIters));
  if (retval != 1)
    die("could not read param file: maxIters", __LINE__, __FILE__);

  /* read in the parameter density */
  retval = fscanf(fp, "density: %f\n", &(params->density));
  if (retval != 1)
    die("could not read param file: density", __LINE__, __FILE__);

  /* read in the parameter viscosity */
  retval = fscanf(fp, "viscosity: %f\n", &(params->viscosity));
  if (retval != 1)
    die("could not read param file: viscosity", __LINE__, __FILE__);

  /* read in the parameter velocity */
  retval = fscanf(fp, "velocity: %f\n", &(params->velocity));
  if (retval != 1)
    die("could not read param file: velocity", __LINE__, __FILE__);

  /* read in the parameter type */
  retval = fscanf(fp, "type: %d\n", &(params->type));
  if (retval != 1) die("could not read param file: type", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /* calculation of relaxtion parameter */
  params->omega = 1. / (3. * params->viscosity + 0.5);

  /* Check the calculation stability */
  if (params->velocity > 0.2)
    printf(
        "Warning: There maybe computational instability due to "
        "compressibility.\n");
  if ((2 - params->omega) < 0.15)
    printf("Warning: Possible divergence of results due to relaxation time.\n");

  /* Allocate memory. */

  /* main grid */
  cells_ptr->speeds_0 =
      (t_speed_0 *)malloc(sizeof(t_speed_0) * (params->ny * params->nx));
  if (cells_ptr->speeds_0 == NULL)
    die("cannot allocate memory for cells", __LINE__, __FILE__);

  cells_ptr->speeds_1_8 = (t_speed_1_8 *)aligned_alloc(
      32, sizeof(t_speed_1_8) * (params->ny * params->nx));
  if (cells_ptr->speeds_1_8 == NULL)
    die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  tmp_cells_ptr->speeds_0 =
      (t_speed_0 *)malloc(sizeof(t_speed_0) * (params->ny * params->nx));
  if (tmp_cells_ptr->speeds_0 == NULL)
    die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  tmp_cells_ptr->speeds_1_8 = (t_speed_1_8 *)aligned_alloc(
      32, sizeof(t_speed_1_8) * (params->ny * params->nx));
  if (tmp_cells_ptr->speeds_1_8 == NULL)
    die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));
  if (*obstacles_ptr == NULL)
    die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density / 9.f;
  float w2 = params->density / 36.f;

  for (int jj = 0; jj < params->ny; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      /* centre */
      cells_ptr->speeds_0[ii + jj * params->nx] = w0;
      /* axis directions */
      cells_ptr->speeds_1_8[ii + jj * params->nx].speeds_1_8[0] = w1;
      cells_ptr->speeds_1_8[ii + jj * params->nx].speeds_1_8[1] = w1;
      cells_ptr->speeds_1_8[ii + jj * params->nx].speeds_1_8[2] = w1;
      cells_ptr->speeds_1_8[ii + jj * params->nx].speeds_1_8[3] = w1;
      /* diagonals */
      cells_ptr->speeds_1_8[ii + jj * params->nx].speeds_1_8[4] = w2;
      cells_ptr->speeds_1_8[ii + jj * params->nx].speeds_1_8[5] = w2;
      cells_ptr->speeds_1_8[ii + jj * params->nx].speeds_1_8[6] = w2;
      cells_ptr->speeds_1_8[ii + jj * params->nx].speeds_1_8[7] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      (*obstacles_ptr)[ii + jj * params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    /* Center the obstacle on the y-axis. */
    yy = yy + params->ny / 2;

    /* some checks */
    if (retval != 3)
      die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1)
      die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1)
      die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1)
      die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy * params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /* allocate space to hold the velocity of the cells at the inlet. */
  *inlets_ptr = (float *)malloc(sizeof(float) * params->ny);

  return EXIT_SUCCESS;
}

/* finalise, including freeing up allocated memory */
int finalise(const t_param *params, t_speed *cells_ptr, t_speed *tmp_cells_ptr,
             int **obstacles_ptr, float **inlets) {
  /*
  ** free up allocated memory
  */
  free(cells_ptr->speeds_0);
  cells_ptr->speeds_0 = NULL;
  free(cells_ptr->speeds_1_8);
  cells_ptr->speeds_1_8 = NULL;

  free(tmp_cells_ptr->speeds_0);
  tmp_cells_ptr->speeds_0 = NULL;
  free(tmp_cells_ptr->speeds_1_8);
  tmp_cells_ptr->speeds_1_8 = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*inlets);
  *inlets = NULL;

  return EXIT_SUCCESS;
}

/* write state of current grid */
int write_state(char *filename, const t_param params, t_speed *cells,
                int *obstacles) {
  FILE *fp; /* file pointer */

  fp = fopen(filename, "w");

  if (fp == NULL) {
    printf("%s\n", filename);
    die("could not open file output file", __LINE__, __FILE__);
  }

  float *tmp_u = (float *)malloc(sizeof(float) * params.nx * params.ny);

  /* loop on grid to calculate the velocity of each cell */
#if defined(LBM_ENV_AUTOLAB)
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, obstacles, fp, tmp_u) \
    num_threads(8)
#else
#pragma omp parallel for default(none) \
    shared(params, cells, obstacles, fp, tmp_u) num_threads(8)
#endif
#else
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(cells, obstacles, fp, tmp_u) \
    num_threads(omp_get_num_procs())
#else
#pragma omp parallel for default(none) shared( \
        params, cells, obstacles, fp, tmp_u) num_threads(omp_get_num_procs())
#endif
#endif
  for (int jj = 0; jj < params.ny; jj++) {
    float local_density; /* per grid cell sum of densities */
    float u_x;           /* x-component of velocity in grid cell */
    float u_y;           /* y-component of velocity in grid cell */
    float u;             /* norm--root of summed squares--of u_x and u_y */

    float local_density_1_2 = 0.f;
    float local_density_3_4 = 0.f;
    float local_density_5_6 = 0.f;
    float local_density_7_8 = 0.f;

    int index = 0;
    for (int ii = 0; ii < params.nx; ii++) {
      index = ii + jj * params.nx;
      if (obstacles[index]) { /* an obstacle cell */
        u = -0.05f;
      } else { /* no obstacle */
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

        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
      }

      tmp_u[ii + jj * params.nx] = u;
    }
  }

  /* write to file */
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* write to file */
      fprintf(fp, "%d %d %.12E\n", ii, jj, tmp_u[ii + jj * params.nx]);
    }
  }

  /* close file */
  fclose(fp);

  free(tmp_u);

  return EXIT_SUCCESS;
}