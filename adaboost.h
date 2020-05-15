#ifndef __adaboost_H__
#define __adaboost_H__ 

#include <math.h>
#include "omp.h"
#include <time.h>
#include "calloc_errchk.h"
#include "bit_op.h"

void *calloc_errchk(size_t, size_t, const char *);

int adaboost_calloc(const unsigned long T,		    
		    const unsigned long N,
		    const unsigned long dim,
		    double **w,
		    double **p,
		    double **err,
		    unsigned long **lernerAxsis,
		    int **lernerPred,
		    double **beta){

  *lernerAxsis = calloc_errchk(T, sizeof(unsigned long), "calloc lernerAxsis");
  *lernerPred = calloc_errchk(T, sizeof(int), "calloc lernerPred");
  *w = calloc_errchk(N, sizeof(double), "calloc w");
  *p = calloc_errchk(N, sizeof(double), "calloc p");
  *beta = calloc_errchk(T, sizeof(double), "calloc beta");
  *err = calloc_errchk(dim, sizeof(double), "calloc count");

  return 0;
}
		           	    
int adaboost_learn(const unsigned int **x, 
		   const int *y,
		   const unsigned long T,
		   const unsigned long N,
		   const unsigned long dim,
		   unsigned long **lernerAxis,
		   int **lernerPred,
		   double **beta){
  // clock_t begin = clock();

  double *w, *p, *err;
  double wsum, epsilon, min, max;

  unsigned long i, t, d, argmind, argmaxd;
  adaboost_calloc(T, N, dim, &w, &p, &err, lernerAxis, lernerPred, beta);
  omp_set_num_threads(omp_get_max_threads());
  double initial_value = 1.0 / N;
    #pragma omp parallel for
    for(i = 0; i < N; i++){
      w[i] = initial_value;
    }
  for(t = 0; t < T; t++){
    /* step 1 : compute normalized weights p[] */
    {
      wsum = 0;
      #pragma omp parallel for reduction(+: wsum)
      for(i = 0; i < N; i++){
	      wsum += w[i];
      }
      double invert_wsum = 1.0 / wsum;
      #pragma omp parallel for
      for(i = 0; i < N; i++) {
	      p[i] = w[i] * invert_wsum;
      }
    }
    /* step 2 : find the most appropriate weak lerner */
    {
      #pragma omp parallel for
      for(d = 0; d < dim; d++){
      	err[d] = 0;
      }
      #pragma omp parallel for collapse(2)
      for(i = 0; i < N; i++){
	      for(d = 0; d < dim; d++){
	       if(get_bit(x[i], d)!= y[i]){
	         err[d] += p[i];
	        }
	      }
      }
    {
	  max = min = err[0];
	  argmaxd = argmind = 0;
    // #pragma omp parallel for reduction(min: min)
    // 	for(d = 1; d < dim; d++){
	  //     if(err[d] < min){
	  //       min = err[d];
	  //       argmind = d;
	  //     }
    //   }
    // #pragma omp parallel for reduction(max: max)
    //   for (d = 1; d < dim; d++) {
    //     if(err[d] > max){
	  //       max = err[d];
	  //       argmaxd = d;
	  //     }
    //   }
      #pragma omp parallel for reduction(max: max) reduction(min: min)
	    for(d = 1; d < dim; d++){
	      if(err[d] < min){
	        min = err[d];
	        argmind = d;
	      }else if(err[d] > max){
	        max = err[d];
	        argmaxd = d;
	      }
      }
    }
    {
	  if(max + min > 1.0){
	    (*lernerAxis)[t] = argmaxd;
	    (*lernerPred)[t] = 1;
	    epsilon = 1 - max;
	  }else{
	    (*lernerAxis)[t] = argmind;
	    (*lernerPred)[t] = 0;
	    epsilon = min;       
	  }
    }
    }
    /* step 3: compute new weithgts */
    {
      (*beta)[t] = epsilon / (1 - epsilon);
      int lernerPredict = (*lernerPred)[t];
      int lernerAxisNum = (*lernerAxis)[t];
      double betaT = (*beta)[t];
      
      #pragma omp parallel for
      for (i = 0; i < N; i++) {
        if ((lernerPredict == 0 && get_bit(x[i], lernerAxisNum) == y[i]) || (lernerPredict == 1 && get_bit(x[i], lernerAxisNum) == y[i])) {
          w[i] *= betaT;
        }
      }
  //     for(i = 0; i < N; i++){
	// if(((*lernerPred)[t] == 0 && get_bit(x[i], (*lernerAxis)[t]) == y[i]) ||
	//    ((*lernerPred)[t] == 1 && get_bit(x[i], (*lernerAxis)[t]) != y[i])){
	//   w[i] *= (*beta)[t];
	//   }
  //   }
    }
  }
  free(w);
  free(p);
  free(err);
  // clock_t end = clock();
  // double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  // printf("time spent is %f\n", time_spent);
  return 0;
}

int adaboost_apply(const unsigned long *lernerAxis,
		   const int *lernerPred,
		   const double *beta,
		   const unsigned long T,
		   const unsigned long N,
		   const unsigned int **x,
		   int **pred){
  // clock_t begin = clock();
  unsigned int i, t;
  double threshold = 0, sum;
  {
    for(t = 0; t < T; t++){
      threshold -= log(beta[t]);
    }
    threshold /= 2;
  }
  for(i = 0; i < N; i++){
    sum = 0;
    for(t = 0; t < T; t++){
      if(lernerPred[t] != get_bit(x[i], lernerAxis[t])){
	sum -= log(beta[t]);
      }
    }
    (*pred)[i] = ((sum >= threshold) ? 1 : 0);
  }
  // clock_t end = clock();
  // double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  // printf("time spent for apply is %f\n", time_spent);
  return 0;
}

#endif
