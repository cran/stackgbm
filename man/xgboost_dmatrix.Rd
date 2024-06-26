% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/wrappers_xgboost.R
\name{xgboost_dmatrix}
\alias{xgboost_dmatrix}
\title{Create xgb.DMatrix object}
\usage{
xgboost_dmatrix(data, label = NULL, ...)
}
\arguments{
\item{data}{Matrix or file.}

\item{label}{Labels (optional).}

\item{...}{Additional parameters.}
}
\value{
An \code{xgb.DMatrix} object.
}
\description{
Create xgb.DMatrix object
}
\examples{
\dontshow{if (is_installed_xgboost()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
sim_data <- msaenet::msaenet.sim.binomial(
  n = 100,
  p = 10,
  rho = 0.6,
  coef = rnorm(5, mean = 0, sd = 10),
  snr = 1,
  p.train = 0.8,
  seed = 42
)

xgboost_dmatrix(sim_data$x.tr, label = sim_data$y.tr)
xgboost_dmatrix(sim_data$x.te)
\dontshow{\}) # examplesIf}
}
