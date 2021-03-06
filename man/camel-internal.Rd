\name{camel-internal}
\alias{tiger.likelihood}
\alias{tiger.tracel2}
\alias{camel.tiger.cv}
\alias{part.cv}
\alias{camel.tiger.clime.mfista}
\alias{camel.tiger.slasso.mfista}
\alias{camel.slim.lad.mfista}
\alias{camel.slim.sqrt.mfista}
\alias{camel.slim.dantzig.mfista}
\alias{camel.cmr.mfista}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Internal camel functions}
\description{
  Internal camel functions
}
\usage{
tiger.likelihood(Sigma, Omega)
tiger.tracel2(Sigma, Omega)
camel.tiger.cv(obj, loss=c("likelihood", "tracel2"), fold=5)
part.cv(n, fold)
camel.tiger.clime.mfista(Sigma, d, maxdf, mu, lambda, shrink, prec, max.ite)
camel.tiger.slasso.mfista(data, n, d, maxdf, mu, lambda, shrink, prec, max.ite)
camel.slim.lad.mfista(Y, X, lambda, nlambda, n, d, maxdf, mu, max.ite, prec, 
                      intercept, verbose)
camel.slim.sqrt.mfista(Y, X, lambda, nlambda, n, d, maxdf, mu, max.ite, prec, 
                       intercept, verbose)
camel.slim.dantzig.mfista(Y, X, lambda, nlambda, n, d, maxdf, mu, max.ite, prec, 
                          intercept, verbose)
camel.cmr.mfista(Y, X, lambda, nlambda, n, d, m, mu, max.ite, prec)

}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Sigma}{Covariance matrix.}
  \item{Omega}{Inverse covariance matrix.}
  \item{obj}{An object with S3 class returned from \code{"tiger"}.}
  \item{loss}{Type of loss function for cross validation.}
  \item{fold}{The number of fold for cross validatio.}
  \item{n}{The number of observations (sample size).}
  \item{d}{Dimension of data.}
  \item{m}{Columns of parameters in multivariate regression.}
  \item{maxdf}{Maximal degree of freedom.}
  \item{lambda}{Grid of non-negative values for the regularization parameter lambda.}
  \item{nlambda}{The number of the regularization parameter lambda.}
  \item{shrink}{Shrinkage of regularization parameter based on precision of estimation.}
  \item{mu}{The smooth surrogate parameter. }
  \item{prec}{Stopping criterion.}
  \item{max.ite}{Maximal value of iterations.}
  \item{data}{\code{n} by \code{d} data matrix. }
  \item{Y}{Dependent variables in linear regression. }
  \item{X}{Design matrix in linear regression. }
  \item{intercept}{Whether the intercept is included in the model.}
  \item{verbose}{Tracing information printing is disabled if \code{verbose = FALSE}.}
}
\details{
  These are not intended for use by users.   
}

%\references{
%}

\author{
Xingguo Li, Tuo Zhao, and Han Liu \cr
Maintainer: Xingguo Li <xingguo.leo@gmail.com>
}
%\note{
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{camel.tiger}}, \code{\link{camel.slim}}, \code{\link{camel.cmr}} and \code{\link{camel-package}}.
}
