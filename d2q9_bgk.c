#include "d2q9_bgk.h"
#include <stdio.h>
#include <string.h>


/* The main processes in one step */
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int streaming(const t_param params, t_speed* cells, t_speed* tmp_cells);
int obstacle(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int boundary(const t_param params, t_speed* cells, t_speed* tmp_cells, float* inlets);

int aligned_collision(const t_param params, aligned_t_speed* cells, aligned_t_speed* tmp_cells, int* obstacles);
int aligned_streaming(const t_param params, aligned_t_speed* cells, aligned_t_speed* tmp_cells);
int aligned_obstacle(const t_param params, aligned_t_speed* cells, aligned_t_speed* tmp_cells, int* obstacles);
int aligned_boundary(const t_param params, aligned_t_speed* cells, aligned_t_speed* tmp_cells, float* inlets);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** collision(), obstacle(), streaming() & boundary()
*/
// int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, float* inlets, int* obstacles)
// {
//   /* The main time overhead, you should mainly optimize these processes. */
//   collision(params, cells, tmp_cells, obstacles);
//   obstacle(params, cells, tmp_cells, obstacles);
//   streaming(params, cells, tmp_cells);
//   boundary(params, cells, tmp_cells, inlets);
//   return EXIT_SUCCESS;
// }

int aligned_timestep(const t_param params, aligned_t_speed*cells, aligned_t_speed*tmp_cells, float *inlets, int *obstacles)
{
  /* The main time overhead, you should mainly optimize these processes. */
  aligned_collision(params, cells, tmp_cells, obstacles);
  // aligned_obstacle(params, cells, tmp_cells, obstacles);
  aligned_streaming(params, cells, tmp_cells);
  aligned_boundary(params, cells, tmp_cells, inlets);
  return EXIT_SUCCESS;
}




/*
** The collision of fluids in the cell is calculated using 
** the local equilibrium distribution and relaxation process
*/
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles) {
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */

  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */

#pragma omp parallel for num_threads(NUM_THREADS)
  for (int jj = 0; jj < params.ny; jj++)
  {
  for (int ii = 0; ii < params.nx; ii++)  
    {
      if (!obstacles[ii + jj*params.nx]){
        /* compute local density total */
        float local_density = 0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk]; 
        }
        // __m256 ymm = _mm256_loadu_ps(cells[ii + jj*params.nx].speeds + 1);
        // __m256 ymm2 = _mm256_permute2f128_ps(ymm , ymm , 1);
        // ymm = _mm256_add_ps(ymm, ymm2);
        // ymm = _mm256_hadd_ps(ymm, ymm);
        // ymm = _mm256_hadd_ps(ymm, ymm);
        // local_density = _mm256_cvtss_f32(ymm) + local_density;
        /* compute x velocity component */
        float u_x = (cells[ii + jj*params.nx].speeds[1]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[8]
                      - (cells[ii + jj*params.nx].speeds[3]
                         + cells[ii + jj*params.nx].speeds[6]
                         + cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        const __m256 cells_vec = _mm256_loadu_ps(cells[ii + jj*params.nx].speeds + 1);
        // const __m256 cells_w = _mm256_setr_ps(1,0,-1,0,1,-1,-1,1);
        // ymm = _mm256_mul_ps(cells_vec, cells_w);
        // ymm2 = _mm256_permute2f128_ps(ymm , ymm , 1);
        // ymm = _mm256_add_ps(ymm, ymm2);
        // ymm = _mm256_hadd_ps(ymm, ymm);
        // ymm = _mm256_hadd_ps(ymm, ymm);
        // float u_x = _mm256_cvtss_f32(ymm) / local_density;

        /* compute y velocity component */
        float u_y = (cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[6]
                      - (cells[ii + jj*params.nx].speeds[4]
                         + cells[ii + jj*params.nx].speeds[7]
                         + cells[ii + jj*params.nx].speeds[8]))
                     / local_density;
        // const __m256 cells_w2 = _mm256_setr_ps(0,1,0,-1,1,1,-1,-1);
        // ymm = _mm256_mul_ps(cells_vec, cells_w2);
        // ymm2 = _mm256_permute2f128_ps(ymm , ymm , 1);
        // ymm = _mm256_add_ps(ymm, ymm2);
        // ymm = _mm256_hadd_ps(ymm, ymm);
        // ymm = _mm256_hadd_ps(ymm, ymm);
        // float u_y = _mm256_cvtss_f32(ymm) / local_density;

        float u_sq = u_x * u_x + u_y * u_y; 
        /* directional velocity components */
        /* directional velocity components */

        /* equilibrium densities */
        // float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */

        // d_equ[0] = w0 * local_density * (1.f - u_sq / (2.f * c_sq));
        const float c0 = local_density * (1.f - u_sq / (2.f * c_sq));
        tmp_cells[ii + jj*params.nx].speeds[0] = (1-params.omega) * cells[ii + jj*params.nx].speeds[0] + c0 * w0 * params.omega;

        const __m256 x = _mm256_setr_ps(u_x,u_y,-u_x,-u_y,u_x+u_y,-u_x+u_y,-u_x-u_y,u_x-u_y);
        const __m256 w_vec = _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2);
        
        // tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0]
        //                           + params.omega
        //                           * (w0 * local_density * (1.f - u_sq / (2.f * c_sq)) - cells[ii + jj*params.nx].speeds[0]);

        //const __m256 d_equ_vec = _mm256_mul_ps(w_vec, _mm256_mul_ps(_mm256_set1_ps(local_density), _mm256_add_ps(_mm256_set1_ps(1.f), _mm256_add_ps(_mm256_div_ps(u_vec, _mm256_set1_ps(c_sq)), _mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(u_vec, u_vec), _mm256_mul_ps(_mm256_set1_ps(2.f), _mm256_mul_ps(_mm256_set1_ps(c_sq), _mm256_set1_ps(c_sq)))), _mm256_div_ps(_mm256_set1_ps(u_sq), _mm256_mul_ps(_mm256_set1_ps(2.f), _mm256_set1_ps(c_sq))))))));
        const __m256 t0 = _mm256_set1_ps(c0);
        const __m256 t1 = _mm256_set1_ps(local_density / c_sq);
        const __m256 t2 = _mm256_set1_ps(local_density / (2 * c_sq * c_sq));
        const __m256 d_equ_vec = _mm256_mul_ps(w_vec,_mm256_fmadd_ps(x, _mm256_fmadd_ps(x, t2 , t1) , t0));
        
        
        /* relaxation step */

        // const __m256 cells_vec = _mm256_loadu_ps(cells[ii + jj*params.nx].speeds + 1); 
        const __m256 omega_vec = _mm256_set1_ps(params.omega);
        const __m256 temp_vec2 = _mm256_fmadd_ps(omega_vec, _mm256_sub_ps(d_equ_vec, cells_vec), cells_vec);
        
        _mm256_storeu_ps(tmp_cells[ii + jj*params.nx].speeds + 1, temp_vec2);
      }
    }
  }
  return EXIT_SUCCESS;
}

/*
** For obstacles, mirror their speed.
*/
int aligned_obstacle(const t_param params, aligned_t_speed* cells, aligned_t_speed* tmp_cells, int* obstacles) {

  /* loop over the cells in the grid */
#pragma omp parallel for num_threads(NUM_THREADS)
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after collision, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells->stay[ii + jj*params.nx] = cells->stay[ii + jj*params.nx];
        tmp_cells->other[ii + jj*params.nx].speeds[1 - 1] = cells->other[ii + jj*params.nx].speeds[3 - 1];
        tmp_cells->other[ii + jj*params.nx].speeds[2 - 1] = cells->other[ii + jj*params.nx].speeds[4 - 1];
        tmp_cells->other[ii + jj*params.nx].speeds[3 - 1] = cells->other[ii + jj*params.nx].speeds[1 - 1];
        tmp_cells->other[ii + jj*params.nx].speeds[4 - 1] = cells->other[ii + jj*params.nx].speeds[2 - 1];
        tmp_cells->other[ii + jj*params.nx].speeds[5 - 1] = cells->other[ii + jj*params.nx].speeds[7 - 1];
        tmp_cells->other[ii + jj*params.nx].speeds[6 - 1] = cells->other[ii + jj*params.nx].speeds[8 - 1];
        tmp_cells->other[ii + jj*params.nx].speeds[7 - 1] = cells->other[ii + jj*params.nx].speeds[5 - 1];
        tmp_cells->other[ii + jj*params.nx].speeds[8 - 1] = cells->other[ii + jj*params.nx].speeds[6 - 1];
      }
    }
  }
  return EXIT_SUCCESS;
}

/*
** The collision of fluids in the cell is calculated using 
** the local equilibrium distribution and relaxation process
*/
int aligned_collision(const t_param params, aligned_t_speed* cells, aligned_t_speed* tmp_cells, int* obstacles) {
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */
  const __m256 w_vec = _mm256_setr_ps(w1, w1, w1, w1, w2, w2, w2, w2);
  const __m256 omega_vec = _mm256_set1_ps(params.omega);
  const __m256 c_sq_n1 = _mm256_set1_ps(1 / c_sq);
  const __m256 c_sq_n2_2 = _mm256_set1_ps(1. / 2 / c_sq / c_sq);
  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */

#pragma omp parallel for num_threads(NUM_THREADS)
  for (int jj = 0; jj < params.ny; jj++)
  {
    float local_density, u_x, u_y, u_sq,c0;
    float ret[8];
  for (int ii = 0; ii < params.nx; ii++)  
    {
      if (!obstacles[ii + jj*params.nx]){
        /* compute local density total */
        local_density = cells->stay[ii + jj*params.nx];
        __m256 t0,t1,t2,d_equ_vec,temp_vec2,x,cells_vec;
        for (int kk = 0; kk < NSPEEDS - 1; kk++)
        {
          local_density += cells->other[ii + jj*params.nx].speeds[kk]; 
        }
        u_x = (cells -> other[ii + jj*params.nx].speeds[1 - 1]
                      + cells -> other[ii + jj*params.nx].speeds[5 - 1]
                      + cells -> other[ii + jj*params.nx].speeds[8 - 1]
                      - (cells -> other[ii + jj*params.nx].speeds[3 - 1]
                         + cells -> other[ii + jj*params.nx].speeds[6 - 1]
                         + cells -> other[ii + jj*params.nx].speeds[7 - 1]))
                     / local_density;
        u_y = (cells -> other[ii + jj*params.nx].speeds[2 - 1]
                      + cells -> other[ii + jj*params.nx].speeds[5 - 1]
                      + cells -> other[ii + jj*params.nx].speeds[6 - 1]
                      - (cells -> other[ii + jj*params.nx].speeds[4 - 1]
                         + cells -> other[ii + jj*params.nx].speeds[7 - 1]
                         + cells -> other[ii + jj*params.nx].speeds[8 - 1]))
                     / local_density;

        u_sq = u_x * u_x + u_y * u_y; 
        c0 = local_density * (1.f - u_sq / (2.f * c_sq));
        const __m256 ld = _mm256_set1_ps(local_density);
        t0 = _mm256_set1_ps((1.f - u_sq / (2.f * c_sq)));
        
        x = _mm256_setr_ps(u_x,u_y,-u_x,-u_y,u_x+u_y,-u_x+u_y,-u_x-u_y,u_x-u_y);
        cells_vec = _mm256_loadu_ps(cells -> other[ii + jj*params.nx].speeds);
        
        t0 = t0;
        t1 = c_sq_n1; 
        t2 = c_sq_n2_2;
        
        // d_equ_vec = _mm256_mul_ps(w_vec,_mm256_add_ps(t0,_mm256_mul_ps(x,_mm256_add_ps(t1,_mm256_mul_ps(x,t2)))));
        d_equ_vec = _mm256_mul_ps(_mm256_mul_ps(w_vec, ld),_mm256_fmadd_ps(x, _mm256_fmadd_ps(x, t2 , t1) , t0));
        
        // temp_vec2 = _mm256_add_ps(cells_vec, _mm256_mul_ps(omega_vec, _mm256_sub_ps(d_equ_vec, cells_vec)));
        temp_vec2 = _mm256_fmadd_ps(omega_vec, _mm256_sub_ps(d_equ_vec, cells_vec), cells_vec);
        _mm256_storeu_ps(ret, temp_vec2);
        tmp_cells->stay[ii + jj*params.nx] = (1-params.omega) * cells->stay[ii + jj*params.nx] + c0 * w0 * params.omega;
      }else{
        /* called after collision, so taking values from scratch space
        ** mirroring, and writing into main grid */
        // memcpy(cells->other[ii + jj*params.nx].speeds, tmp_cells->other[ii + jj*params.nx].speeds, sizeof(float) * 8);
        tmp_cells->stay[ii + jj*params.nx] = cells->stay[ii + jj*params.nx];
        ret[0] = cells->other[ii + jj*params.nx].speeds[2];
        ret[1] = cells->other[ii + jj*params.nx].speeds[3];
        ret[2] = cells->other[ii + jj*params.nx].speeds[0];
        ret[3] = cells->other[ii + jj*params.nx].speeds[1];
        ret[4] = cells->other[ii + jj*params.nx].speeds[6];
        ret[5] = cells->other[ii + jj*params.nx].speeds[7];
        ret[6] = cells->other[ii + jj*params.nx].speeds[4];
        ret[7] = cells->other[ii + jj*params.nx].speeds[5];
      }
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);
      tmp_cells->other[x_e + jj * params.nx].speeds[0] = ret[0]; 
      tmp_cells->other[x_w + jj * params.nx].speeds[2] = ret[2]; 

      if (jj == 0) {
        y_s = 1;
        tmp_cells->other[ii + jj*params.nx].speeds[7] = ret[5];
        tmp_cells->other[ii + jj*params.nx].speeds[3] = ret[1];
        tmp_cells->other[ii + jj*params.nx].speeds[6] = ret[4];
        tmp_cells->other[x_w + y_s*params.nx].speeds[6] = ret[6];
        tmp_cells->other[ii + y_s*params.nx].speeds[3] = ret[3];
        tmp_cells->other[x_e + y_s*params.nx].speeds[7] = ret[7];
      } else if (jj == params.ny - 1) {
        y_n = params.ny - 2;
        tmp_cells->other[x_w + y_n*params.nx].speeds[5] = ret[5];
        tmp_cells->other[ii + y_n*params.nx].speeds[1] = ret[1];
        tmp_cells->other[x_e + y_n*params.nx].speeds[4] = ret[4];
        tmp_cells->other[ii + jj*params.nx].speeds[4] = ret[6];
        tmp_cells->other[ii + jj*params.nx].speeds[1] = ret[3];
        tmp_cells->other[ii + jj*params.nx].speeds[5] = ret[7];
      } else {
        y_s = jj + 1;
        y_n = jj - 1;
        tmp_cells->other[x_w + y_n*params.nx].speeds[5] = ret[5];
        tmp_cells->other[ii + y_n*params.nx].speeds[1] = ret[1];
        tmp_cells->other[x_e + y_n*params.nx].speeds[4] = ret[4];
        tmp_cells->other[x_w + y_s*params.nx].speeds[6] = ret[6];
        tmp_cells->other[ii + y_s*params.nx].speeds[3] = ret[3];
        tmp_cells->other[x_e + y_s*params.nx].speeds[7] = ret[7];
      }
  // int y_n = jj + 1;
  // int y_s = jj - 1;
  // int x_e = ii + 1;
  // int x_w = ii - 1;
  // if(x_e < params.nx)
  //   tmp_cells->other[x_e + jj*params.nx].speeds[1 - 1] = ret[0];
  // if(y_n < params.ny)
  //   tmp_cells->other[ii + y_n*params.nx].speeds[2 - 1] = ret[1];
  // if(x_w >= 0)
  //   tmp_cells->other[x_w + jj*params.nx].speeds[3 - 1] = ret[2];
  // if(y_s >= 0)
  //   tmp_cells->other[ii + y_s*params.nx].speeds[4 - 1] = ret[3];
  // if(x_e < params.nx && y_n < params.ny)
  //   tmp_cells->other[x_e + y_n*params.nx].speeds[5 - 1] = ret[4];
  // if(x_w >= 0 && y_n < params.ny)
  //   tmp_cells->other[x_w + y_n*params.nx].speeds[6 - 1] = ret[5];
  // if(x_w >= 0 && y_s>= 0)
  //   tmp_cells->other[x_w + y_s*params.nx].speeds[7 - 1] = ret[6];
  // if(x_e < params.nx && y_s >= 0)
  //   tmp_cells->other[x_e + y_s*params.nx].speeds[8 - 1] = ret[7];
  // // top wall (bounce)
  // jj = params.ny -1;
  // for(ii = 0; ii < params.nx; ii++){
  //   cells->other[ii + jj*params.nx].speeds[4 - 1] = tmp_cells->other[ii + jj*params.nx].speeds[2 - 1];
  //   cells->other[ii + jj*params.nx].speeds[7 - 1] = tmp_cells->other[ii + jj*params.nx].speeds[5 - 1];
  //   cells->other[ii + jj*params.nx].speeds[8 - 1] = tmp_cells->other[ii + jj*params.nx].speeds[6 - 1];
  // }

  // // bottom wall (bounce)
  // jj = 0;
  // for(ii = 0; ii < params.nx; ii++){
  //   cells->other[ii + jj*params.nx].speeds[2 - 1] = tmp_cells->other[ii + jj*params.nx].speeds[4 - 1];
  //   cells->other[ii + jj*params.nx].speeds[5 - 1] = tmp_cells->other[ii + jj*params.nx].speeds[7 - 1];
  //   cells->other[ii + jj*params.nx].speeds[6 - 1] = tmp_cells->other[ii + jj*params.nx].speeds[8 - 1];
  // }
    }
  }
  
  return EXIT_SUCCESS;
}




int aligned_streaming(const t_param params, aligned_t_speed* cells, aligned_t_speed* tmp_cells) {
  /* loop over _all_ cells */

// #pragma omp parallel for num_threads(NUM_THREADS)
//   for (int jj = 0; jj < params.ny; jj++)
//   {
//     for (int ii = 0; ii < params.nx; ii++)
//     {
//       /* determine indices of axis-direction neighbours
//       ** respecting periodic boundary conditions (wrap around) */
//       int y_n = (jj + 1) % params.ny;
//       int x_e = (ii + 1) % params.nx;
//       int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);
//       int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);
//       /* propagate densities from neighbouring cells, following
//       ** appropriate directions of travel and writing into
//       ** scratch space grid */

//       cells->other[ii + jj * params.nx].speeds[2 - 1] = tmp_cells -> other[ii + y_s * params.nx].speeds[2 - 1]; /* south */
//       cells->other[ii + jj * params.nx].speeds[5 - 1] = tmp_cells -> other[x_w + y_s * params.nx].speeds[5 - 1]; /* north-east */
//       cells->other[ii + jj * params.nx].speeds[6 - 1] = tmp_cells -> other[x_e + y_s * params.nx].speeds[6 - 1]; /* north-west */
//       cells->other[ii + jj * params.nx].speeds[1 - 1] = tmp_cells -> other[x_w + jj * params.nx].speeds[1 - 1]; /* east */
//       cells->other[ii + jj * params.nx].speeds[3 - 1] = tmp_cells -> other[x_e + jj * params.nx].speeds[3 - 1]; /* west */
//       cells->other[ii + jj * params.nx].speeds[4 - 1] = tmp_cells -> other[ii + y_n * params.nx].speeds[4 - 1]; /* south */
//       cells->other[ii + jj * params.nx].speeds[8 - 1] = tmp_cells -> other[x_w + y_n * params.nx].speeds[8 - 1]; /* south-east */
//       cells->other[ii + jj * params.nx].speeds[7 - 1] = tmp_cells -> other[x_e + y_n * params.nx].speeds[7 - 1]; /* south-west */
    
//     }
//   }
  
  // swap cells.stay and tmp_cells.stay
  void* tmp;
  
  //print cells->stay 's location
  // printf("cells->stay's location is %p\n", cells->stay);
  // printf("tmp_cells->stay's location is %p\n", tmp_cells->stay);
  tmp = cells->stay;
  cells->stay = tmp_cells->stay;
  tmp_cells->stay = tmp;

  tmp = cells->other;
  cells->other = tmp_cells->other;
  tmp_cells->other = tmp;
  
  // printf("After swap, cells->stay's location is %p\n", cells->stay);
  // printf("After swap, tmp_cells->stay's location is %p\n", tmp_cells->stay);
  // printf("-----------------------------\n");
  // memcpy(cells->stay, tmp_cells->stay, params.nx * params.ny * sizeof(float));
  // memset(tmp_cells->stay, 0, params.nx * params.ny * sizeof(float));
  return EXIT_SUCCESS;
}
int aligned_boundary(const t_param params, aligned_t_speed* cells,  aligned_t_speed* tmp_cells, float* inlets) {
  /* Set the constant coefficient */
  const float cst1 = 2.0/3.0;
  const float cst2 = 1.0/6.0;
  const float cst3 = 1.0/2.0;

  int ii, jj; 
  float local_density;
  
  // // top wall (bounce)
  // jj = params.ny -1;
  // for(ii = 0; ii < params.nx; ii++){
  //   cells->other[ii + jj*params.nx].speeds[4 - 1] = tmp_cells->other[ii + jj*params.nx].speeds[2 - 1];
  //   cells->other[ii + jj*params.nx].speeds[7 - 1] = tmp_cells->other[ii + jj*params.nx].speeds[5 - 1];
  //   cells->other[ii + jj*params.nx].speeds[8 - 1] = tmp_cells->other[ii + jj*params.nx].speeds[6 - 1];
  // }

  // // bottom wall (bounce)
  // jj = 0;
  // for(ii = 0; ii < params.nx; ii++){
  //   cells->other[ii + jj*params.nx].speeds[2 - 1] = tmp_cells->other[ii + jj*params.nx].speeds[4 - 1];
  //   cells->other[ii + jj*params.nx].speeds[5 - 1] = tmp_cells->other[ii + jj*params.nx].speeds[7 - 1];
  //   cells->other[ii + jj*params.nx].speeds[6 - 1] = tmp_cells->other[ii + jj*params.nx].speeds[8 - 1];
  // }

  // left wall (inlet)
  ii = 0;
  for(jj = 0; jj < params.ny; jj++){
    local_density = ( cells->stay[ii + jj*params.nx]
                      + cells->other[ii + jj*params.nx].speeds[2 - 1]
                      + cells->other[ii + jj*params.nx].speeds[4 - 1]
                      + 2.0 * cells->other[ii + jj*params.nx].speeds[3 - 1]
                      + 2.0 * cells->other[ii + jj*params.nx].speeds[6 - 1]
                      + 2.0 * cells->other[ii + jj*params.nx].speeds[7 - 1]
                      )/(1.0 - inlets[jj]);

    cells->other[ii + jj*params.nx].speeds[1 - 1] = cells->other[ii + jj*params.nx].speeds[3 - 1]
                                        + cst1*local_density*inlets[jj];

    cells->other[ii + jj*params.nx].speeds[5 - 1] = cells->other[ii + jj*params.nx].speeds[7 - 1]
                                        - cst3*(cells->other[ii + jj*params.nx].speeds[2 - 1]-cells->other[ii + jj*params.nx].speeds[4 - 1])
                                        + cst2*local_density*inlets[jj];

    cells->other[ii + jj*params.nx].speeds[8 - 1] = cells->other[ii + jj*params.nx].speeds[6 - 1]
                                        + cst3*(cells->other[ii + jj*params.nx].speeds[2 - 1]-cells->other[ii + jj*params.nx].speeds[4 - 1])
                                        + cst2*local_density*inlets[jj];
  
  }

  // right wall (outlet)
  ii = params.nx-1;
  for(jj = 0; jj < params.ny; jj++){
    for (int kk = 0; kk < NSPEEDS - 1; kk++)
    {
      cells->other[ii + jj*params.nx].speeds[kk] = cells->other[ii-1 + jj*params.nx].speeds[kk];
    }
    cells->stay[ii + jj*params.nx] = cells->stay[ii-1 + jj*params.nx];
  }
  
  return EXIT_SUCCESS;
}
/*
** Work with boundary conditions. The upper and lower boundaries use the rebound plane, 
** the left border is the inlet of fixed speed, and 
** the right border is the open outlet of the first-order approximation.
*/
int boundary(const t_param params, t_speed* cells,  t_speed* tmp_cells, float* inlets) {
  /* Set the constant coefficient */
  const float cst1 = 2.0/3.0;
  const float cst2 = 1.0/6.0;
  const float cst3 = 1.0/2.0;

  int ii, jj; 
  float local_density;
  
  // top wall (bounce)
  jj = params.ny -1;
  for(ii = 0; ii < params.nx; ii++){
    cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii + jj*params.nx].speeds[2];
    cells[ii + jj*params.nx].speeds[7] = tmp_cells[ii + jj*params.nx].speeds[5];
    cells[ii + jj*params.nx].speeds[8] = tmp_cells[ii + jj*params.nx].speeds[6];
  }

  // bottom wall (bounce)
  jj = 0;
  for(ii = 0; ii < params.nx; ii++){
    cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii + jj*params.nx].speeds[4];
    cells[ii + jj*params.nx].speeds[5] = tmp_cells[ii + jj*params.nx].speeds[7];
    cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii + jj*params.nx].speeds[8];
  }

  // left wall (inlet)
  ii = 0;
  for(jj = 0; jj < params.ny; jj++){
    local_density = ( cells[ii + jj*params.nx].speeds[0]
                      + cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[4]
                      + 2.0 * cells[ii + jj*params.nx].speeds[3]
                      + 2.0 * cells[ii + jj*params.nx].speeds[6]
                      + 2.0 * cells[ii + jj*params.nx].speeds[7]
                      )/(1.0 - inlets[jj]);

    cells[ii + jj*params.nx].speeds[1] = cells[ii + jj*params.nx].speeds[3]
                                        + cst1*local_density*inlets[jj];

    cells[ii + jj*params.nx].speeds[5] = cells[ii + jj*params.nx].speeds[7]
                                        - cst3*(cells[ii + jj*params.nx].speeds[2]-cells[ii + jj*params.nx].speeds[4])
                                        + cst2*local_density*inlets[jj];

    cells[ii + jj*params.nx].speeds[8] = cells[ii + jj*params.nx].speeds[6]
                                        + cst3*(cells[ii + jj*params.nx].speeds[2]-cells[ii + jj*params.nx].speeds[4])
                                        + cst2*local_density*inlets[jj];
  
  }

  // right wall (outlet)
  ii = params.nx-1;
  for(jj = 0; jj < params.ny; jj++){

    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      cells[ii + jj*params.nx].speeds[kk] = cells[ii-1 + jj*params.nx].speeds[kk];
    }
    
  }
  
  return EXIT_SUCCESS;
}



/*
** Particles flow to the corresponding cell according to their speed direaction.
*/
int streaming(const t_param params, t_speed* cells, t_speed* tmp_cells) {
  /* loop over _all_ cells */

#pragma omp parallel for num_threads(NUM_THREADS)
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      cells[ii + jj * params.nx].speeds[2] = tmp_cells[ii + y_s * params.nx].speeds[2]; /* south */
      cells[ii + jj * params.nx].speeds[5] = tmp_cells[x_w + y_s * params.nx].speeds[5]; /* north-east */
      cells[ii + jj * params.nx].speeds[6] = tmp_cells[x_e + y_s * params.nx].speeds[6]; /* north-west */
      
      cells[ii + jj * params.nx].speeds[0] = tmp_cells[ii + jj * params.nx].speeds[0]; /* central cell, no movement */
      cells[ii + jj * params.nx].speeds[1] = tmp_cells[x_w + jj * params.nx].speeds[1]; /* east */
      cells[ii + jj * params.nx].speeds[3] = tmp_cells[x_e + jj * params.nx].speeds[3]; /* west */

      cells[ii + jj * params.nx].speeds[4] = tmp_cells[ii + y_n * params.nx].speeds[4]; /* south */
      cells[ii + jj * params.nx].speeds[8] = tmp_cells[x_w + y_n * params.nx].speeds[8]; /* south-east */
      cells[ii + jj * params.nx].speeds[7] = tmp_cells[x_e + y_n * params.nx].speeds[7]; /* south-west */
    
    }
  }

  return EXIT_SUCCESS;
}

/*
** For obstacles, mirror their speed.
*/
int obstacle(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles) {

  /* loop over the cells in the grid */
#pragma omp parallel for num_threads(NUM_THREADS)
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after collision, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0];
        tmp_cells[ii + jj*params.nx].speeds[1] = cells[ii + jj*params.nx].speeds[3];
        tmp_cells[ii + jj*params.nx].speeds[2] = cells[ii + jj*params.nx].speeds[4];
        tmp_cells[ii + jj*params.nx].speeds[3] = cells[ii + jj*params.nx].speeds[1];
        tmp_cells[ii + jj*params.nx].speeds[4] = cells[ii + jj*params.nx].speeds[2];
        tmp_cells[ii + jj*params.nx].speeds[5] = cells[ii + jj*params.nx].speeds[7];
        tmp_cells[ii + jj*params.nx].speeds[6] = cells[ii + jj*params.nx].speeds[8];
        tmp_cells[ii + jj*params.nx].speeds[7] = cells[ii + jj*params.nx].speeds[5];
        tmp_cells[ii + jj*params.nx].speeds[8] = cells[ii + jj*params.nx].speeds[6];
      }
    }
  }
  return EXIT_SUCCESS;
}