library(MASS)

log_post <- function(beta, X, y, sigma2=10){
  p <- 1 / (1 + exp(-X %*% beta))
  sum(y*log(p) + (1-y)*log(1-p)) - sum(beta^2)/(2*sigma2)
}

gibbs_sampler <- function(X, y, n_iter=5000){
  beta <- rep(0, ncol(X))
  samples <- matrix(0, n_iter, length(beta))
  for (i in 1:n_iter){
    proposal <- beta + rnorm(length(beta), 0, 0.1)
    log_alpha <- log_post(proposal, X, y) - log_post(beta, X, y)
    if (log(runif(1)) < log_alpha) beta <- proposal
    samples[i,] <- beta
  }
  colMeans(samples[(n_iter/2):n_iter,])
}
