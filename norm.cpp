#include <Rcpp.h>
#include <omp.h>
using namespace Rcpp;

// [[Rcpp::plugins(openmp)]]

// [[Rcpp::export]]
NumericMatrix dist_for(const NumericVector & x, int N, int M) {
  
  // NumericVector ans(x.size(), 0.0);
  // unsigned int N = x.nrow(), M = x.ncol();
  const double * rowi = (const double *) &x[0];
  const double * rowj = (const double *) &x[0];
  NumericMatrix ans(N, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < i; ++j) {
      ans(i, j) = 0.0;
      for (int k = 0; k < M; ++k)
        ans(i, j) += pow(rowi[i + k * N] - rowj[j + k * N], 2.0);
      ans(i, j) = sqrt(ans(i, j));
      ans(j, i) = ans(i, j);
    }
  
  return ans;
  
}

// [[Rcpp::export]]
NumericMatrix dist_simd(const NumericVector & x, int N, int M, int ncores = 4) {
  
  
  omp_set_num_threads(ncores);
  
  // NumericVector ans(x.size(), 0.0);
  // unsigned int N = x.nrow(), M = x.ncol();
  NumericMatrix ans(N, N); 
  const double * rowi = (const double *) &x[0];
  const double * rowj = (const double *) &x[0];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < i; ++j) {
      double tmp = 0.0;
      ans(i, j) = 0.0;
#pragma omp simd reduction(+:tmp) 
      for (int k = 0; k < M; ++k)
        tmp += pow(rowi[i + k * N] - rowj[j + k * N], 2.0);
      ans(i, j) = sqrt(tmp);
      ans(j, i) = ans(i, j);
    }
  
  return ans;
  
}

// [[Rcpp::export]]
NumericMatrix dist_omp_simd(const NumericVector & x, int N, int M, int ncores = 4) {
  
  omp_set_num_threads(ncores);
  
  // NumericVector ans(x.size(), 0.0);
  // unsigned int N = x.nrow(), M = x.ncol();
  NumericMatrix ans(N, N); 
  const double * rowi = (const double *) &x[0];
  const double * rowj = (const double *) &x[0];
#pragma omp parallel for shared(rowi,rowj,N,M) collapse(1)
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < i; ++j) {
      double tmp = 0.0;
      ans(i, j) = 0.0;
#pragma omp simd reduction(+:tmp) 
      for (int k = 0; k < M; ++k)
        tmp += pow(rowi[i + k * N] - rowj[j + k * N], 2.0);
      ans(i, j) = sqrt(tmp);
      ans(j, i) = ans(i, j);
    }
    
    return ans;
  
}

// [[Rcpp::export]]
NumericMatrix dist_omp(const NumericVector & x, int N, int M, int ncores = 4) {
  
  omp_set_num_threads(ncores);
  
  // NumericVector ans(x.size(), 0.0);
  // unsigned int N = x.nrow(), M = x.ncol();
  NumericMatrix ans(N, N); 
  const double * rowi = (const double *) &x[0];
  const double * rowj = (const double *) &x[0];
#pragma omp parallel for shared(rowi,rowj,N,M) collapse(1)
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < i; ++j) {
      ans(i, j) = 0.0;
      for (int k = 0; k < M; ++k)
        ans(i, j) += pow(rowi[i + k * N] - rowj[j + k * N], 2.0);
      ans(i, j) = sqrt(ans(i,j));
      ans(j, i) = ans(i, j);
    }
    
    return ans;
  
}

