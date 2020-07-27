
# Rcpp + OpenMP

This repository shows how much speed gains can be obtained from using
[OpenMP](https://openmp.org), and in particular, the `omp simd` and
`parallel for` instructions.

The test consists on computing the pair-wise distances between rows in a
matrix of size `N` by `M`. The equivalent function in R is `dist()`, and
here we redefined it using
[`Rcpp`](https://cran.r-project.org/package=Rcpp). The program is

The file [norm.cpp](norm.cpp) contains the C++ source code for the dist
functions. The compiles function are:

  - `dist_omp_simd` Using the pragma directives `parallel for` and
    `simd`.

  - `dist_omp` Using the pragma `parallel for`.

  - `dist_simd` Using the pragma `simd`.

  - `dist_for` no directives.

  - `dist_for_arma2` Using Armadillo with vectorized functions.

  - `dist_for_arma1` Armadillo implementation with for-loops.

## Speed benchmark

Whe using multicore, I’m only using 2 cores

``` r
Sys.setenv("PKG_CXXFLAGS"=paste("-fopenmp -O2 -mavx2 -march=core-avx2 -mtune=core-avx2"))
Rcpp::sourceCpp("norm.cpp")

library(microbenchmark)
set.seed(718243)
N <- 500
M <- 1000
x <- matrix(runif(N * M), nrow = N)

(ans_bm <- microbenchmark(
  `SIMD + parfor` = dist_omp_simd(x, N, M, 2),
  `parfor`        = dist_omp(x, N, M, 2),
  `SIMD`          = dist_simd(x, N, M),
  `serial`        = dist_for(x, N, M),
  `arma sugar`    = dist_for_arma2(x,N,M),
  `arma`          = dist_for_arma1(x,N,M),
  R               = as.matrix(dist(x)),
  times           = 10,
  unit            = "relative"
))
```

    ## Unit: relative
    ##           expr       min        lq      mean    median        uq       max
    ##  SIMD + parfor  1.000000  1.000000  1.000000  1.000000  1.000000  1.000000
    ##         parfor  4.525767  4.228444  3.956531  3.741240  3.754374  3.870529
    ##           SIMD  1.998500  1.986318  1.924191  1.752240  1.722760  2.122128
    ##         serial  6.187921  5.770530  5.179253  4.993561  4.923769  4.139801
    ##     arma sugar  8.582125  8.084118  7.641437  7.085500  6.739811  7.963009
    ##           arma 17.258725 16.340381 15.588027 15.380473 15.175025 13.925154
    ##              R 14.175100 13.198820 12.506306 11.462402 11.972508 12.457738
    ##  neval
    ##     10
    ##     10
    ##     10
    ##     10
    ##     10
    ##     10
    ##     10

As a reference, the elapsed time in ms for R and SIMD + parfor is

    ## Unit: milliseconds
    ##           expr       min        lq      mean    median        uq       max
    ##  SIMD + parfor  28.62259  31.20543  35.82479  36.48984  38.97481  47.33635
    ##              R 405.72800 411.87482 448.03582 418.26125 466.62621 589.70381
    ##  neval
    ##     10
    ##     10

Overall, in my machine, the SIMD+parfor combo outperforms all the
others. Let’s see if the results are equivalent. At the very least, we
should only observe small differences (if any) b/c of precision:

``` r
Rcpp::sourceCpp("norm.cpp")
ans0 <- as.matrix(dist(x))
ans1a <- dist_omp_simd(x, N, M)
ans1b <- dist_omp(x, N, M)
ans1c <- dist_simd(x, N, M)
ans1d <- dist_for(x, N, M)
range(ans0 - ans1b)
```

    ## [1] -1.071123e-07  9.346249e-08

``` r
range(ans1a - ans1b)
```

    ## [1] 0.000000e+00 5.151435e-14

``` r
range(ans1b - ans1c)
```

    ## [1] -9.346248e-08  1.071123e-07

``` r
range(ans1c - ans1d)
```

    ## [1] -2.842171e-14  3.197442e-14

The programs were compiled on a machine with an [Intel(R) Core(TM)
i5-7200U CPU @ 2.50GHz
processor](https://ark.intel.com/content/www/us/en/ark/products/95443/intel-core-i5-7200u-processor-3m-cache-up-to-3-10-ghz.html)
which works with [AVX2
instructions](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2),
i.e. we can literally vectorize 4 operations at a time (on top of
multi-threading). One important thing to consider is that for this to
work we had to generate a copy of the R matrix into a double vector so
that elements were contiguous (which is important for SIMD).

Finally, the
[`microbenchmark`](https://cran.r-project.org/package=microbenchmark) R
package offers a nice viz with boxplot comparing all the methods:

``` r
op <- par(mai = par("mai") * c(2,1,1,1))
boxplot(ans_bm, las = 2, xlab = "")
```

![](README_files/figure-gfm/viz-1.png)<!-- -->

``` r
par(op)
```

## Session info

``` r
sessionInfo()
```

    ## R version 4.0.2 (2020-06-22)
    ## Platform: x86_64-pc-linux-gnu (64-bit)
    ## Running under: Ubuntu 18.04.4 LTS
    ## 
    ## Matrix products: default
    ## BLAS:   /usr/lib/x86_64-linux-gnu/atlas/libblas.so.3.10.3
    ## LAPACK: /usr/lib/x86_64-linux-gnu/atlas/liblapack.so.3.10.3
    ## 
    ## locale:
    ##  [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
    ##  [3] LC_TIME=en_US.UTF-8        LC_COLLATE=en_US.UTF-8    
    ##  [5] LC_MONETARY=en_US.UTF-8    LC_MESSAGES=en_US.UTF-8   
    ##  [7] LC_PAPER=en_US.UTF-8       LC_NAME=C                 
    ##  [9] LC_ADDRESS=C               LC_TELEPHONE=C            
    ## [11] LC_MEASUREMENT=en_US.UTF-8 LC_IDENTIFICATION=C       
    ## 
    ## attached base packages:
    ## [1] stats     graphics  grDevices utils     datasets  methods   base     
    ## 
    ## other attached packages:
    ## [1] microbenchmark_1.4-7
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] compiler_4.0.2            magrittr_1.5             
    ##  [3] tools_4.0.2               htmltools_0.5.0          
    ##  [5] RcppArmadillo_0.9.900.1.0 yaml_2.2.1               
    ##  [7] Rcpp_1.0.5                codetools_0.2-16         
    ##  [9] stringi_1.4.6             rmarkdown_2.3            
    ## [11] knitr_1.29                stringr_1.4.0            
    ## [13] xfun_0.15                 digest_0.6.25            
    ## [15] rlang_0.4.6               evaluate_0.14
