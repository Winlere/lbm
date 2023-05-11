#include "types.h"
#include "utils.h"
#include <time.h>


/* utility functions */
void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void* aligned_malloc(size_t required_bytes, size_t alignment)
{
    void* p1; // original block
    void** p2; // aligned block
    int offset = alignment - 1 + sizeof(void*);
    if ((p1 = (void*)malloc(required_bytes + offset)) == NULL)
    {
       return NULL;
    }
    p2 = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
}

void aligned_free(void *p)
{
    free(((void**)p)[-1]);
}

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** inlets_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
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
  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  /* read in the parameter density */
  retval = fscanf(fp, "density: %f\n", &(params->density));
  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  /* read in the parameter viscosity */
  retval = fscanf(fp, "viscosity: %f\n", &(params->viscosity));
  if (retval != 1) die("could not read param file: viscosity", __LINE__, __FILE__);

  /* read in the parameter velocity */
  retval = fscanf(fp, "velocity: %f\n", &(params->velocity));
  if (retval != 1) die("could not read param file: velocity", __LINE__, __FILE__);

  /* read in the parameter type */
  retval = fscanf(fp, "type: %d\n", &(params->type));
  if (retval != 1) die("could not read param file: type", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /* calculation of relaxtion parameter */
  params->omega=1./(3.*params->viscosity+0.5);

  /* Check the calculation stability */
  if(params->velocity>0.2)
    printf("Warning: There maybe computational instability due to compressibility.\n");
  if((2-params->omega) < 0.15)
    printf("Warning: Possible divergence of results due to relaxation time.\n");

  /* Allocate memory. */

  /* main grid */
  *cells_ptr = (t_speed*)aligned_malloc(sizeof(t_speed) * (params->ny * params->nx),64);


  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)aligned_malloc(sizeof(t_speed) * (params->ny * params->nx),64);
  
  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = aligned_malloc(sizeof(int) * (params->ny * params->nx),64);
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* Center the obstacle on the y-axis. */
    yy = yy + params->ny/2;
    
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /* allocate space to hold the velocity of the cells at the inlet. */
  *inlets_ptr = (float*)aligned_malloc(sizeof(float) * params->ny,64);

  return EXIT_SUCCESS;
}
/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise_aligned(const char* paramfile, const char* obstaclefile, t_param* params, aligned_t_speed* cells_ptr, aligned_t_speed* tmp_cells_ptr, int** obstacles_ptr, float** inlets_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
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
  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  /* read in the parameter density */
  retval = fscanf(fp, "density: %f\n", &(params->density));
  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  /* read in the parameter viscosity */
  retval = fscanf(fp, "viscosity: %f\n", &(params->viscosity));
  if (retval != 1) die("could not read param file: viscosity", __LINE__, __FILE__);

  /* read in the parameter velocity */
  retval = fscanf(fp, "velocity: %f\n", &(params->velocity));
  if (retval != 1) die("could not read param file: velocity", __LINE__, __FILE__);

  /* read in the parameter type */
  retval = fscanf(fp, "type: %d\n", &(params->type));
  if (retval != 1) die("could not read param file: type", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /* calculation of relaxtion parameter */
  params->omega=1./(3.*params->viscosity+0.5);

  /* Check the calculation stability */
  if(params->velocity>0.2)
    printf("Warning: There maybe computational instability due to compressibility.\n");
  if((2-params->omega) < 0.15)
    printf("Warning: Possible divergence of results due to relaxation time.\n");

  /* Allocate memory. */

  /* main grid */
  cells_ptr -> stay = (float*)aligned_malloc(sizeof(float) * (params->ny * params->nx),64);
  cells_ptr -> other = (t_speed_8*)aligned_malloc(sizeof(t_speed_8) * (params->ny * params->nx),64);


  // if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  tmp_cells_ptr -> stay = (float*)aligned_malloc(sizeof(float) * (params->ny * params->nx),64);
  tmp_cells_ptr -> other = (t_speed_8*)aligned_malloc(sizeof(t_speed_8) * (params->ny * params->nx),64);
  
  // if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = aligned_malloc(sizeof(int) * (params->ny * params->nx),64);
  // if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      cells_ptr -> stay[ii + jj*params->nx] = w0;
      /* axis directions */
      cells_ptr -> other[ii + jj*params->nx].speeds[1 - 1] = w1;
      cells_ptr -> other[ii + jj*params->nx].speeds[2 - 1] = w1;
      cells_ptr -> other[ii + jj*params->nx].speeds[3 - 1] = w1;
      cells_ptr -> other[ii + jj*params->nx].speeds[4 - 1] = w1;
      /* diagonals -> other */ 
      cells_ptr -> other[ii + jj*params->nx].speeds[5 - 1] = w2;
      cells_ptr -> other[ii + jj*params->nx].speeds[6 - 1] = w2;
      cells_ptr -> other[ii + jj*params->nx].speeds[7 - 1] = w2;
      cells_ptr -> other[ii + jj*params->nx].speeds[8 - 1] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* Center the obstacle on the y-axis. */
    yy = yy + params->ny/2;
    
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /* allocate space to hold the velocity of the cells at the inlet. */
  *inlets_ptr = (float*)aligned_malloc(sizeof(float) * params->ny,64);

  return EXIT_SUCCESS;
}

int write_state_aligned(char *filename, const t_param params, aligned_t_speed cells, int *obstacles){
    FILE* fp;                    /* file pointer */
  float local_density;         /* per grid cell sum of densities */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(filename, "w");

  if (fp == NULL)
  {
    printf("%s\n",filename);
    die("could not open file output file", __LINE__, __FILE__);
  }

  /* loop on grid to calculate the velocity of each cell */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      if (obstacles[ii + jj*params.nx])
      { /* an obstacle cell */
        u = -0.05f; 
      }
      else
      { /* no obstacle */
        local_density = cells . stay[ii + jj*params.nx];

        for (int kk = 0; kk < NSPEEDS - 1; kk++)
        {
          local_density += cells . other[ii + jj*params.nx].speeds[kk];
        }
        
        /* compute x velocity component */
        u_x = (cells . other [ii + jj*params.nx].speeds[1 - 1]
               + cells . other [ii + jj*params.nx].speeds[5 - 1]
               + cells . other [ii + jj*params.nx].speeds[8 - 1]
               - (cells . other [ii + jj*params.nx].speeds[3 - 1]
                  + cells . other [ii + jj*params.nx].speeds[6 - 1]
                  + cells . other [ii + jj*params.nx].speeds[7 - 1]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells . other [ii + jj*params.nx].speeds[2 - 1] 
               + cells . other [ii + jj*params.nx].speeds[5 - 1]
               + cells . other [ii + jj*params.nx].speeds[6 - 1]
               - (cells . other [ii + jj*params.nx].speeds[4 - 1]
                  + cells . other [ii + jj*params.nx].speeds[7 - 1]
                  + cells . other [ii + jj*params.nx].speeds[8 - 1]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E\n", ii, jj, u);
    }
  }
  
  /* close file */
  fclose(fp);

  return EXIT_SUCCESS;
}

int finalise_aligned(const t_param *params, aligned_t_speed cells_ptr, aligned_t_speed tmp_cells_ptr, int **obstacles_ptr, float **inlets){
    aligned_free(cells_ptr.other);
    aligned_free(cells_ptr.stay);
    cells_ptr.other = NULL;
    cells_ptr.stay = NULL;
    
    
    

    aligned_free(tmp_cells_ptr .other);
    aligned_free(tmp_cells_ptr .stay);
    tmp_cells_ptr .other = NULL;
    tmp_cells_ptr .stay = NULL;
    

    aligned_free(*obstacles_ptr);
    *obstacles_ptr = NULL;

    aligned_free(*inlets);
    *inlets = NULL;

    return EXIT_SUCCESS;
}

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** inlets)
{
  /*
  ** free up allocated memory
  */
  aligned_free(*cells_ptr);
  *cells_ptr = NULL;

  aligned_free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  aligned_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  aligned_free(*inlets);
  *inlets = NULL;

  return EXIT_SUCCESS;
}


/* write state of current grid */
int write_state(char* filename, const t_param params, t_speed* cells, int* obstacles)
{
  FILE* fp;                    /* file pointer */
  float local_density;         /* per grid cell sum of densities */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(filename, "w");

  if (fp == NULL)
  {
    printf("%s\n",filename);
    die("could not open file output file", __LINE__, __FILE__);
  }

  /* loop on grid to calculate the velocity of each cell */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      if (obstacles[ii + jj*params.nx])
      { /* an obstacle cell */
        u = -0.05f; 
      }
      else
      { /* no obstacle */
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }
        
        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[8]
               - (cells[ii + jj*params.nx].speeds[3]
                  + cells[ii + jj*params.nx].speeds[6]
                  + cells[ii + jj*params.nx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[6]
               - (cells[ii + jj*params.nx].speeds[4]
                  + cells[ii + jj*params.nx].speeds[7]
                  + cells[ii + jj*params.nx].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E\n", ii, jj, u);
    }
  }
  
  /* close file */
  fclose(fp);

  return EXIT_SUCCESS;
}