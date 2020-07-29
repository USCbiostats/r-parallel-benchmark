#include <RcppArmadillo.h>
#include <omp.h>
using namespace Rcpp;

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat dist_for_arma1(const arma::mat & x, int N, int M) {
  
  arma::mat ans(N, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < i; ++j) {
      ans(i, j) = 0.0;
      for (int k = 0; k < M; ++k)
        ans(i, j) += pow(x(i, k) - x(j, k), 2.0);
      ans(i, j) = sqrt(ans(i, j));
      ans(j, i) = ans(i, j);
    }
    
    return ans;
  
}

// [[Rcpp::export]]
arma::mat dist_for_arma2(const arma::mat & x, int N, int M) {
  
  arma::mat ans(N, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < i; ++j) {
      ans(i,j) = sqrt(
        arma::accu(arma::pow(x.row(i) - x.row(j), 2.0))
      );
      ans(j, i) = ans(i, j);
    }
    
    return ans;
  
}

// [[Rcpp::export]]
NumericMatrix dist_for(const NumericMatrix & x, int N, int M) {
  
  double x_[N*M];
  int counter = 0;
  for (int i = 0; i < N; ++i)
    for(int j = 0; j < M; ++j)
      x_[counter++] = x(i,j);
  
  NumericMatrix ans(N, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < i; ++j) {
      ans(i, j) = 0.0;
      for (int k = 0; k < M; ++k)
        ans(i, j) += pow(x_[i*M + k] - x_[j*M + k], 2.0);
      ans(i, j) = sqrt(ans(i, j));
      ans(j, i) = ans(i, j);
    }
  
  return ans;
  
}

// [[Rcpp::export]]
NumericMatrix dist_simd(const NumericMatrix & x, int N, int M, int ncores = 4) {
  
  omp_set_num_threads(ncores);
  
  double x_[N*M];
  int counter = 0;
  for (int i = 0; i < N; ++i)
    for(int j = 0; j < M; ++j)
      x_[counter++] = x(i,j);
  
  // NumericVector ans(x.size(), 0.0);
  // unsigned int N = x.nrow(), M = x.ncol();
  NumericMatrix ans(N, N); 
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < i; ++j) {
      double tmp = 0.0;
      ans(i, j) = 0.0;
#pragma omp simd reduction(+:tmp) 
      for (int k = 0; k < M; ++k)
        tmp += pow(x_[i*M + k] - x_[j*M + k], 2.0);
      ans(i, j) = sqrt(tmp);
      ans(j, i) = ans(i, j);
    }
  
  return ans;
  
}



// [[Rcpp::export]]
NumericMatrix dist_omp_simd(const NumericMatrix & x, int N, int M, int ncores = 4) {
  
  omp_set_num_threads(ncores);
  
  float x_[N*M];
  int counter = 0;
  for (int i = 0; i < N; ++i)
    for(int j = 0; j < M; ++j)
      x_[counter++] = x(i,j);
  
  NumericMatrix ans(N,N);
#pragma omp parallel for shared(x_,N,M) collapse(1)
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < i; ++j) {
      double tmp = 0.0;
#pragma omp simd reduction(+:tmp) simdlen(4) 
      for (int k = 0; k < M; ++k)
        tmp += pow(x_[i*M + k] - x_[j*M + k], 2.0);
      ans(i,j) = sqrt(tmp);
      ans(j,i) = ans(i,j);
    }
  }
  
  return ans;
  
}

// [[Rcpp::export]]
NumericMatrix dist_omp_simd_ptr(const NumericMatrix & x, int N, int M, int ncores = 4) {
  
  omp_set_num_threads(ncores);
  
  const double * x_ = (const double *) &x[0];
  
  NumericMatrix ans(N,N);
#pragma omp parallel for shared(x_,N,M) collapse(1)
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < i; ++j) {
      double tmp = 0.0;
#pragma omp simd reduction(+:tmp) simdlen(4) 
      for (int k = 0; k < M; ++k)
        tmp += pow(x_[i*M + k] - x_[j*M + k], 2.0);
      ans(i,j) = sqrt(tmp);
      ans(j,i) = ans(i,j);
    }
  }
  
  return ans;
  
}

// [[Rcpp::export]]
NumericMatrix dist_omp(const NumericMatrix & x, int N, int M, int ncores = 4) {
  
  omp_set_num_threads(ncores);
  
  float x_[N*M];
  int counter = 0;
  for (int i = 0; i < N; ++i)
    for(int j = 0; j < M; ++j)
      x_[counter++] = x(i,j);
  NumericMatrix ans(N, N); 
  
#pragma omp parallel for shared(x_,N,M) collapse(1)
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < i; ++j) {
      ans(i, j) = 0.0;
      for (int k = 0; k < M; ++k)
        ans(i, j) += pow(x_[i*M + k] - x_[j*M + k], 2.0);
      ans(i, j) = sqrt(ans(i,j));
      ans(j, i) = ans(i, j);
    }
  }
    
    return ans;
  
}

