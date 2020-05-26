#include <Rcpp.h>
#include <omp.h>
using namespace Rcpp;

// [[Rcpp::plugins(openmp)]]

// [[Rcpp::export]]
int add_sugar(NumericVector x, NumericVector y, NumericVector & ans) {
  
  ans = exp(x + y);
  
  return 1;
  
}

// [[Rcpp::export]]
int add_for(NumericVector x, NumericVector y, NumericVector & ans) {
  
  // NumericVector ans(x.size(), 0.0);
  for (int i = 0; i < (int) x.size(); ++i)
    ans[i] = exp(x[i] + y[i]);
  
  return 1;
  
}

// [[Rcpp::export]]
int add_simd(NumericVector x, NumericVector y, NumericVector & ans,int ncores = 4) {
  
  omp_set_num_threads(ncores);
  
  // NumericVector ans(x.size(), 0.0);
#pragma omp simd
  for (int i = 0; i < (int) x.size(); ++i)
    ans[i] = exp(x[i] + y[i]);
  
  return 1;
  
}

// [[Rcpp::export]]
int add_omp(NumericVector x, NumericVector y, NumericVector & ans, int ncores = 4) {
  
  omp_set_num_threads(ncores);
  
  // NumericVector ans(x.size(), 0.0);
#pragma omp parallel for shared(ans) 
  for (int i = 0; i < (int) x.size(); ++i)
    ans[i] = exp(x[i] + y[i]);
  
  return 1;
  
}

// [[Rcpp::export]]
int add_omp_simd(NumericVector x, NumericVector y, NumericVector & ans, int ncores = 4) {
  
  omp_set_num_threads(ncores);

  // NumericVector ans(x.size(), 0.0);
#pragma omp distribute parallel for simd
  for (int i = 0; i < (int) x.size(); ++i)
    ans[i] = exp(x[i] + y[i]);
  
  return 1;
  
}


/***R

library(microbenchmark)
set.seed(718243)
N <- 2e4
x <- runif(N)
y <- runif(N)
z <- double(N)

ans <- microbenchmark(
  add_simd(x,y,z),
  add_omp(x,y,z),
  add_sugar(x,y,z),
  add_for(x,y,z),
  add_omp_simd(x,y,z),
  times = 1e4,
  unit  = "relative"
)
op <- par(mai = par("mai") * c(2,1,1,1))
boxplot(ans, las =2, xlab = "")
par(op)
ans
*/