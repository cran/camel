\name{camel.tiger}
\alias{camel.tiger}

\title{
Tuning Insensitive Graph Estimation and Regression
}

\description{
The function "camel.cmr" implements TIGER and Calibrated CLIME using L1 norm regularization}

\usage{
camel.tiger(data, lambda = NULL, nlambda = NULL, lambda.min.ratio = NULL, 
      method = "slasso", sym = "or", shrink=NULL, prec = 1e-4, mu = 0.01,
      max.ite = 1e4, standardize = FALSE, correlation = FALSE,
      perturb = TRUE, verbose = TRUE)
}

\arguments{
  \item{data}{There are 2 options for \code{"clime"}: (1) \code{data} is an \code{n} by \code{d} data matrix (2) a \code{d} by \code{d} sample covariance matrix. The program automatically identifies the input matrix by checking the symmetry. (\code{n} is the sample size and \code{d} is the dimension). For \code{"slasso"}, covariance input is not supported.}
  \item{lambda}{A sequence of decresing positive numbers to control the regularization. Typical usage is to leave the input \code{lambda = NULL} and have the program compute its own \code{lambda} sequence based on \code{nlambda} and \code{lambda.min.ratio}. Users can also specify a sequence to override this. Default value is from \eqn{lambda.max} to \code{lambda.min.ratio*lambda.max}. For TIGER, the default value of \eqn{lambda.max} is \eqn{\pi\sqrt{\log(d)/n}}. For CLIME , the default value of \eqn{lambda.max} is the minimum regularization parameter, which yields an all-zero off-diagonal estiamtes.}
  \item{nlambda}{The number of values used in \code{lambda}. Default value is 10.}
  \item{lambda.min.ratio}{The smallest value for \code{lambda}, as a fraction of the uppperbound (\code{MAX}) of the regularization parameter. The program can automatically generate \code{lambda} as a sequence of length = \code{nlambda} starting from \code{MAX} to \code{lambda.min.ratio*MAX} in log scale. The default value is \code{0.25} for TIGER and \code{0.5} for CLIME.}
  \item{method}{TIGER is applied if \code{method = "slasso"}, CLIME is applied if \code{method="clime"}. Default value is \code{"slasso"}.}
  \item{sym}{Symmetrization of output graphs. If \code{sym = "and"}, the edge between node \code{i} and node \code{j} is selected ONLY when both node \code{i} and node \code{j} are selected as neighbors for each other. If \code{sym = "or"}, the edge is selected when either node \code{i} or node \code{j} is selected as the neighbor for each other. The default value is \code{"or"}.}
  \item{shrink}{Shrinkage of regularization parameter based on precision of estimation. The default value is 1.5 if \code{method = "clime"} and the default value is 0 if \code{method="slasso"} or \code{method = "aclime"}.}
  \item{prec}{Stopping criterion. The default value is 1e-4.}
  \item{mu}{The smoothing parameter. The default value is 0.01.}
  \item{max.ite}{The iteration limit. The default value is 1e4.}
  \item{standardize}{All varaibles are standardized to have mean zero and standard deviation one if \code{standardize = TRUE}. The default value is \code{FALSE}.}
  \item{correlation}{Correlation matrix is used as the input of \code{Sigma} for \code{method = "clime"} if \code{correlation = TRUE}. The default value is \code{FALSE}.}
  \item{perturb}{The diagonal of \code{Sigma} is added by a positive value to guarantee that \code{Sigma} is positive definite if \code{perturb = TRUE}. User can specify a numeric value for \code{perturbe}. The default value is TRUE.}
  \item{verbose}{Tracing information is disabled if \code{verbose = FALSE}. The default value is \code{TRUE}.}
  }
\details{
TIGER and Calibrated CLIME adjust the regularization with respect to each column of the sparse precision matrix. Thus it achieves both improved finite sample performance and tuning insensitiveness.
}

\value{
An object with S3 class \code{"tiger"} is returned:  
  \item{data}{
  The \code{n} by \code{d} data matrix or \code{d} by \code{d} sample covariance matrix from the input.
  }
  \item{cov.input}{
  An indicator of the sample covariance. 
  }
  \item{lambda}{
  The sequence of regularization parameters \code{lambda} used in the program.
  }
  \item{nlambda}{
  The number of values used in \code{lambda}.
  }
  \item{icov}{
  A list of \code{d} by \code{d} precision matrices corresponding to regularization parameters.
  }
  \item{sym}{
  The \code{sym} from the input.
  }
  \item{method}{
  The \code{method} from the input. 
  }
  \item{path}{
  A list of \code{d} by \code{d} adjacency matrices of estimated graphs as a graph path corresponding to \code{lambda}.
  }
  \item{sparsity}{
  The sparsity levels of the graph path.
  }
  \item{ite}{
  If \code{method = "clime"}, it is a list of two matrices where ite[[1]] is the number of external iterations and ite[[2]] is the number of internal iterations with the entry of (i,j) as the number of iteration of i-th column and j-th lambda. If \code{method="slasso"}, it is a matrix of iteration with the entry of (i,j) as the number of iteration of i-th column and j-th lambda. 
  }
  \item{df}{
  It is a \code{d} by \code{nlambda} matrix. Each row contains the number of nonzero coefficients along the lasso solution path. 
  }
  \item{standardize}{The \code{standardize} from the input.}
  \item{correlation}{The \code{correlation} from the input.}
  \item{perturb}{The \code{perturb} from the input.}
  \item{verbose}{The \code{verbose} from the input.}
}

\author{
Xingguo Li, Tuo Zhao, and Han Liu \cr
Maintainer: Xingguo Li <xingguo.leo@gmail.com>
}

\references{
1. T. Cai, W. Liu and X. Luo. A constrained L1 minimization approach to sparse precision matrix estimation. \emph{Journal of the American Statistical Association}, 2011. \cr
2. H. Liu and L. Wang. TIGER: A tuning-insensitive approach for optimally estimating large undirected graphs. \emph{Technical Report}, 2012. \cr
}

\seealso{
\code{\link{camel-package}}, \code{\link{camel.tiger.generator}}, \code{\link{camel.tiger.select}}, \code{\link{camel.plot}}, \code{\link{camel.tiger.roc}}, \code{\link{plot.tiger}}, \code{\link{plot.select}}, \code{\link{plot.roc}}, \code{\link{plot.sim}}, \code{\link{print.tiger}}, \code{\link{print.select}}, \code{\link{print.roc}} and \code{\link{print.sim}}.
}


\examples{

## generating data
n = 100
d = 100
D = camel.tiger.generator(n=n,d=d,graph="hub",g=10)
plot(D)

## sparse precision matrix estimation with method "clime"
out1 = camel.tiger(D$data, method = "clime")
plot(out1)
camel.plot(out1$path[[7]])

## sparse precision matrix estimation with method "slasso"
out2 = camel.tiger(D$data, method = "slasso")
plot(out2)
camel.plot(out2$path[[4]])
}
