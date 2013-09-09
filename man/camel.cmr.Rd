\name{camel.cmr}
\alias{camel.cmr}

\title{
Calibrated Multivariate Regression
}

\description{
The function "camel.cmr" implements calibrated multivariate regression using jointly sparse regularization.
}

\usage{
camel.cmr(X, Y, lambda = NULL, nlambda = NULL, prec = 1e-3,
          max.ite = 1e3, mu = 0.01, verbose = TRUE)
}

\arguments{
  \item{Y}{The \eqn{n} by \eqn{m} dimensional response matrix.}
  \item{X}{The \eqn{n} by \eqn{d} design matrix.}
  \item{lambda}{A sequence of decresing positive value to control the regularization. Typical usage is to leave the input \code{lambda = NULL} and have the program compute its own \code{lambda} sequence based on \code{nlambda}, \eqn{d} and \eqn{m}. Users can also specify a sequence to override this. }
  \item{nlambda}{The number of values used in \code{lambda}. Default value is 10.}
  \item{prec}{Stopping criterion. The default value is 1e-3.}
  \item{max.ite}{The iteration limit. The default value is 1e3.}
  \item{mu}{The smoothing parameter. The default value is 0.01.}
  \item{verbose}{Tracing information is disabled if \code{verbose = FALSE}. The default value is \code{TRUE}.}
  }
\details{ 
Calibrated multivariate regression adjusts the regularization with respect to the noise level of each task. Thus it achieves improved statistical performance and the tuning insensitiveness. 
}
\value{
An object with S3 class \code{"camel.cmr"} is returned: 
  \item{beta}{
  A list of matrice of regression estimates where each entry corresponds to a regularization parameter.
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
1. L. Han, L. Wang, and T. Zhao. Multivariate Regression with Calibration. \emph{http://arxiv.org/abs/1305.2238}, 2013.\cr
}

\seealso{
\code{\link{camel-package}}.
}

\examples{
## Generate the design matrix and regression coefficient vector
n = 200
d = 400
m = 13
Sigma = matrix(0.5,d,d)
diag(Sigma) = 1
X = mvrnorm(n,rep(0,d),Sigma)
B = matrix(0,d,m)
B[1,] = 3
B[2,] = 2
B[4,] = 1.5
W = matrix(rnorm(n*m,0,1),n,m)
sig = sqrt(2)
D = sig*diag(2^(c(0:-12)/4))
Z = W\%*\%D
Y = X\%*\%B + Z
out = camel.cmr(X, Y)
}