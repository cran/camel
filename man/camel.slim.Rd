\name{camel.slim}
\alias{camel.slim}

\title{
Calibrated Linear Regression
}

\description{
The function "camel.slime" implements LAD/L1 Lasso, SQRT/L2 Lasso, and carlibrated Dantizg selector using L1 regularization.
}

\usage{
camel.slim(X, Y, lambda = NULL, nlambda = NULL, lambda.min.ratio = NULL,
           method="lq", q = 2, prec = 1e-4, max.ite = 1e4, mu = 0.01,
           intercept = TRUE, verbose = TRUE)
}

\arguments{
  \item{Y}{The \eqn{n} dimensional response vector.}
  \item{X}{The \eqn{n} by \eqn{d} design matrix.}
  \item{lambda}{A sequence of decresing positive value to control the regularization. Typical usage is to leave the input \code{lambda = NULL} and have the program compute its own \code{lambda} sequence based on \code{nlambda} and \code{lambda.min.ratio}. Users can also specify a sequence to override this. Default value is from \eqn{lambda.max} to \code{lambda.min.ratio*lambda.max}. For Lq regression, the default value of \eqn{lambda.max} is \eqn{\pi\sqrt{\log(d)/n}}. For Dantzig selector, the default value of \eqn{lambda.max} is the minimum regularization parameter, which yields an all-zero estiamtes.}
  \item{nlambda}{The number of values used in \code{lambda}. Default value is 5.}
  \item{lambda.min.ratio}{The smallest value for \code{lambda}, as a fraction of the uppperbound (\code{MAX}) of the regularization parameter. The program can automatically generate \code{lambda} as a sequence of length = \code{nlambda} starting from \code{MAX} to \code{lambda.min.ratio*MAX} in log scale. The default value is \code{0.25} for Lq Lasso and \code{0.5} for Dantzig selector.}
  \item{method}{Dantzig selector is applied if \code{method = "dantzig"} and \eqn{L_q} Lasso is applied if \code{method = "lq"}. The default value is \code{"lq"}.}
  \item{q}{The loss function used in Lq Lasso. It is only applicable when \code{method = "lq"} and must be either 1 or 2. The default value is 2.}
  \item{prec}{Stopping criterion. The default value is 1e-4.}
  \item{max.ite}{The iteration limit. The default value is 1e4.}
  \item{mu}{The smoothing parameter. The default value is 0.01.}
  \item{intercept}{Whether the intercept is included in the model. The defulat value is \code{TRUE}.}
  \item{verbose}{Tracing information is disabled if \code{verbose = FALSE}. The default value is \code{TRUE}.}
  }
\details{ 
Calibrated Linear Regression adjust the regularization with respect to the noise level. Thus it achieves both improved finite sample performance and tuning insensitiveness.
}
\value{
An object with S3 class \code{"camel.slim"} is returned: 
  \item{beta}{
  A matrix of regression estimates whose columns correspond to regularization parameters.
  }
  \item{intercept}{The value of intercepts corresponding to regularization parameters. }
  \item{Y}{The value of \code{Y} used in the program. }
  \item{X}{The value of \code{X} used in the program. }
  \item{lambda}{
  The sequence of regularization parameters \code{lambda} used in the program.
  }
  \item{nlambda}{
  The number of values used in \code{lambda}.
  }
  \item{method}{
  The \code{method} from the input. 
  }
  \item{sparsity}{
  The sparsity levels of the solution path.
  }
  \item{ite}{
  A list of vectors where ite[[1]] is the number of external iteration and ite[[2]] is the number of internal iteration with the i-th entry corresponding to the i-th regularization parameter. 
  }
  \item{verbose}{The \code{verbose} from the input.}
}

\author{
Xingguo Li, Tuo Zhao, and Han Liu \cr
Maintainer: Xingguo Li <xingguo.leo@gmail.com>
}

\references{
1. A. Belloni, V. Chernozhukov and L. Wang. Pivotal recovery of sparse signals via conic programming. \emph{Biometrika}, 2012.\cr
2. L. Wang. L1 penalized LAD estimator for high dimensional linear regression. \emph{Journal of Multivariate Analysis}, 2013.\cr
3. E. Candes and T. Tao. The Dantzig selector: Statistical estimation when p is much larger than n. \emph{Annals of Statistics}, 2007.\cr
}

\seealso{
\code{\link{camel-package}}.
}

\examples{
## Generate the design matrix and regression coefficient vector
n = 200
d = 400
X = matrix(rnorm(n*d), n, d)
beta = c(3,2,0,1.5,rep(0,d-4))

## Generate response using Gaussian noise, and fit a sparse linear model using SQRT Lasso
eps.sqrt = rnorm(n)
Y.sqrt = X\%*\%beta + eps.sqrt
out.sqrt = camel.slim(X = X, Y = Y.sqrt, lambda = seq(0.8,0.2,length.out=5))

## Generate response using Cauchy noise, and fit a sparse linear model using LAD Lasso
eps.lad = rt(n = n, df = 1)
Y.lad = X\%*\%beta + eps.lad
out.lad = camel.slim(X = X, Y = Y.lad, q = 1, lambda = seq(0.5,0.2,length.out=5))

## Visualize the solution path
plot(out.sqrt)
plot(out.lad)
}