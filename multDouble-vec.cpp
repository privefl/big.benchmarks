// [[Rcpp::depends(RcppArmadillo, RcppEigen, bigmemory, BH)]]
#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <bigmemory/MatrixAccessor.hpp>

using namespace Rcpp;


/******************************************************************************/

// [[Rcpp::export]]
NumericVector prod1_double(XPtr<BigMatrix> bMPtr, const NumericVector& x) {
  
  MatrixAccessor<double> macc(*bMPtr);
  
  int n = bMPtr->nrow();
  int m = bMPtr->ncol();
  
  NumericVector res(n);
  int i, j;
  
  for (j = 0; j < m; j++) {
    for (i = 0; i < n; i++) {
      res[i] += macc[j][i] * x[j];
    }
  }
  
  return res;
}

/******************************************************************************/

// [[Rcpp::export]]
NumericVector prod4_double(XPtr<BigMatrix> bMPtr, const NumericVector& x) {
  
  MatrixAccessor<double> macc(*bMPtr);
  
  int n = bMPtr->nrow();
  int m = bMPtr->ncol();
  
  NumericVector res(n);
  int i, j;
  
  for (j = 0; j <= m - 4; j += 4) {
    for (i = 0; i < n; i++) { // unrolling optimization
      res[i] += (x[j] * macc[j][i] + x[j+1] * macc[j+1][i]) +
        (x[j+2] * macc[j+2][i] + x[j+3] * macc[j+3][i]);
    } // The parentheses are somehow important. Try without.
  }
  for (; j < m; j++) {
    for (i = 0; i < n; i++) {
      res[i] += x[j] * macc[j][i];
    }
  }
  
  return res;
}

/******************************************************************************/

// [[Rcpp::export]]
Eigen::VectorXd prodEigen_double(XPtr<BigMatrix> bMPtr, 
                                 const Eigen::Map<Eigen::VectorXd> x) {
  
  Eigen::Map<Eigen::MatrixXd> bM((double *)bMPtr->matrix(), 
                                 bMPtr->nrow(), bMPtr->ncol());
  
  return bM * x;
}

/******************************************************************************/

// [[Rcpp::export]]
arma::vec prodArma_double(XPtr<BigMatrix> xpA, const arma::vec& x) {
  
  arma::mat Am((double*) xpA->matrix(), xpA->nrow(), xpA->ncol(), false);
  
  return Am * x;
}

/******************************************************************************/

// [[Rcpp::export]]
arma::vec prodArmaSub_double(XPtr<BigMatrix> xpA, const arma::vec& x,
                             const arma::Row<uint32_t>& ind) {
  arma::mat Am((double *) xpA->matrix(), xpA->nrow(), xpA->ncol(), false);
  
  return Am.rows(ind) * x;
}

/******************************************************************************/
