
# Rcpp + OpenMP

This repository shows how much speed gains can be obtained from using
[OpenMP](https://openmp.org), and in particular, the `omp simd` and
`parallel for` instructions.

The test consists on computing the pair-wise distances between rows in a
matrix of size `N` by `M`. The equivalent function in R is `dist()`, and
here we redefined it using
[`Rcpp`](https://cran.r-project.org/package=Rcpp). The program is

The file [norm.cpp](norm.cpp) contains the C++ source code for the dist
functions.

## Speed benchmark

Whe using multicore, I’m only using 2 cores

``` r
Rcpp::sourceCpp("norm.cpp")

library(microbenchmark)
set.seed(718243)
N <- 200
M <- 1000
x <- matrix(runif(N * M), nrow = N)

(ans_bm <- microbenchmark(
  `SIMD + parfor` = dist_omp_simd(x, N, M, 2),
  `parfor`        = dist_omp(x, N, M, 2),
  `SIMD`          = dist_simd(x, N, M),
  `serial`        = dist_for(x, N, M),
  R               = dist(x),
  times           = 200,
  unit            = "relative"
))
```

    ## Unit: relative
    ##           expr      min       lq     mean   median       uq       max neval
    ##  SIMD + parfor 1.000000 1.000000 1.000000 1.000000 1.000000 1.0000000   200
    ##         parfor 1.139602 1.121602 1.126302 1.115695 1.157925 1.0997163   200
    ##           SIMD 1.322806 1.302088 1.266197 1.296907 1.276461 0.8608174   200
    ##         serial 1.519825 1.491468 1.462143 1.489836 1.484320 0.9928373   200
    ##              R 2.247616 2.213546 2.115620 2.194877 2.101770 1.3271068   200

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

    ## [1] 0 0

``` r
range(ans1a - ans1b)
```

    ## [1] -3.197442e-14  2.842171e-14

``` r
range(ans1b - ans1c)
```

    ## [1] -2.842171e-14  3.197442e-14

``` r
range(ans1c - ans1d)
```

    ## [1] -3.197442e-14  2.842171e-14

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

    ## R version 3.6.3 (2020-02-29)
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
    ##  [1] compiler_3.6.3   magrittr_1.5     tools_3.6.3      htmltools_0.4.0 
    ##  [5] yaml_2.2.0       Rcpp_1.0.3       codetools_0.2-16 stringi_1.4.5   
    ##  [9] rmarkdown_2.1    knitr_1.27       stringr_1.4.0    xfun_0.12       
    ## [13] digest_0.6.23    rlang_0.4.4      evaluate_0.14
