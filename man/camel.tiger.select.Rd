\name{camel.tiger.select}
\alias{camel.tiger.select}

\title{
Model selection for high-dimensional undirected graph estimation
}

\description{
Implements the regularization parameter selection for high dimensional undirected graph estimation. The optional approaches are stability approach to regularization selection (stars) and cross validation selection (cv).
}

\usage{
camel.tiger.select(est, criterion = "stars", stars.subsample.ratio = NULL, 
                   stars.thresh = 0.1,rep.num = 20, fold = 5, 
                   loss="likelihood", verbose = TRUE)
}

\arguments{
  \item{est}{
An object with S3 class \code{"camel.tiger"}
}
  \item{criterion}{
Model selection criterion. \code{"stars"} and \code{"cv"} are available for both graph estimation methods. The default value is \code{"stars"}.
}
  \item{stars.subsample.ratio}{
The subsampling ratio. The default value is \code{10*sqrt(n)/n} when \code{n>144} and \code{0.8} when \code{n<=144}, where \code{n} is the sample size. Only applicable when \code{criterion = "stars"}.
}
  \item{stars.thresh}{
The variability threshold in stars. The default value is \code{0.1}. Only applicable when \code{criterion = "stars"}.
}
  \item{rep.num}{
The number of subsamplings. The default value is \code{20}. 
}
  \item{fold}{
The number of folds used in cross validation. The default value is \code{5}. Only applicable when \code{criterion = "cv"}.
}
  \item{loss}{
  Loss to be used in cross validation. Two losses are available: \code{"likelihood"} and \code{"tracel2"}. Default \code{"likelihood"}. Only applicable when \code{criterion = "cv"}.
  }
  \item{verbose}{
If \code{verbose = FALSE}, tracing information printing is disabled. The default value is \code{TRUE}.
}
}

\details{
Stability approach to regularization selection (stars) is a natural way to select optimal regularization parameter for all three estimation methods. It selects the optimal graph by variability of subsamplings and tends to over-select edges in Gaussian graphical models. Besides selecting the regularization parameters, stars can also provide an additional estimated graph by merging the corresponding subsampled graphs using the frequency counts. The K-fold cross validation is also provided for selecting the parameter \code{lambda}, and two loss functions are adopted as follow
\deqn{
likelihood: Tr(\Sigma \Omega) - \log|\Omega|
}
\deqn{
tracel2: Tr(diag(\Sigma \Omega - I)^2).
}
}

\value{
An object with S3 class "select" is returned:
  \item{refit}{
The optimal graph selected from the graph path
}
  \item{opt.icov}{
The optimal precision matrix selected.
}
  \item{merge}{
The graph path estimated by merging the subsampling paths. Only applicable when the input \code{criterion = "stars"}.
}
  \item{variability}{
The variability along the subsampling paths. Only applicable when the input \code{criterion = "stars"}.
}
  \item{opt.index}{
The index of the selected regularization parameter.
}
  \item{opt.lambda}{
The selected regularization/thresholding parameter.
}
  \item{opt.sparsity}{
The sparsity level of \code{"refit"}.
}

and anything else inluded in the input \code{est}

}

\note{
The model selection is NOT available when the data input is the sample covaraince matrix.
}

\author{
Xingguo Li, Tuo Zhao and Han Liu \cr
Maintainer: Xingguo Li <xingguo.leo@gmail.com>
}

\references{
1. H. Liu and L. Wang. TIGER: A tuning-insensitive approach for optimally estimating large undirected graphs. \emph{Technical Report}, 2012. \cr
2. T. Cai, W. Liu, and X. Luo. A constrained \eqn{\ell_1} minimization approach to sparse precision matrix estimation. \emph{Journal of the American Statistical Association}, 2011.\cr
}

\seealso{
\code{\link{camel.tiger}} and \code{\link{camel-package}}.
}

\examples{
#generate data
L = camel.tiger.generator(d = 20, graph="hub")
out1 = camel.tiger(L$data)

#model selection using stars
out1.select2 = camel.tiger.select(out1, criterion = "stars", stars.thresh = 0.05)
plot(out1.select2)

#model selection using cross validation
out1.select3 = camel.tiger.select(out1, criterion = "cv")
plot(out1.select3)
}
