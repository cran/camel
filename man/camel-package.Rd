\name{camel-package}
\alias{camel-package}
\alias{camel}
\docType{package}
\title{
camel: Calibrated Machine Learning
}
\description{
The package "camel" provides the implementation of a family of high-dimensional calibrated machine learning tools, including (1) LAD, SQRT Lasso and Calibrated Dantzig Selector for estimating sparse linear models; (2) Calibrated Multivariate Regression for estimating sparse multivariate linear models; (3) Tiger, Calibrated Clime for estimating sparse Gaussian graphical models. We adopt the combination of the dual smoothing and monotone fast iterative soft-thresholding algorithm (MFISTA). The computation is memory-optimized using the sparse matrix output, and accelerated by the path following and active set tricks.
}
\details{
\tabular{ll}{
Package: \tab camel\cr
Type: \tab Package\cr
Version: \tab 0.2.0\cr
Date: \tab 2013-09-09\cr
License: \tab GPL-2\cr
}
}

\author{
Xingguo Li, Tuo Zhao, and Han Liu \cr
Maintainer: Xingguo Li <xingguo.leo@gmail.com>
}

\references{
1. A. Belloni, V. Chernozhukov, and L. Wang. Pivotal recovery of sparse signals via conic programming. \emph{Biometrika}, 2012.\cr
2. L. Wang. L1 penalized LAD estimator for high dimensional linear regression. \emph{Journal of Multivariate Analysis}, 2013.\cr
3. E. Candes and T. Tao. The Dantzig selector: Statistical estimation when p is much larger than n. \emph{Annals of Statistics}, 2007.\cr
4. T. Cai, W. Liu, and X. Luo. A constrained \eqn{\ell_1} minimization approach to sparse precision matrix estimation. \emph{Journal of the American Statistical Association}, 2011.\cr
5. H. Liu and L. Wang. TIGER: A tuning-insensitive approach for optimally estimating large undirected graphs. \emph{Technical Report}, 2012. \cr
6. L. Han, L. Wang, and T. Zhao. Multivariate Regression with Calibration. \emph{http://arxiv.org/abs/1305.2238}, 2013.\cr
7. T. Zhao and H. Liu, Sparse Precision Matrix Estimation with Calibration. \emph{Advances in Neural Information Processing systems}, 2013.\cr
8. A. Beck and M. Teboulle. Fast gradient-based algorithms for constrained total variation image denoising and deblurring problems. \emph{IEEE Transactions on Image Processing}, 2009.\cr
9. Y. Nesterov. Smooth minimization of non-smooth functions. \emph{Mathematical Programming}, 2005. \cr
}

\seealso{
\code{\link{camel.tiger}}, \code{\link{camel.slim}} and \code{\link{camel.cmr}} .
}
