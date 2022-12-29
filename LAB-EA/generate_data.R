library(MASS)
round_df <- function(df, digits, max_value, min_value) {
  nums <- vapply(df, is.numeric, FUN.VALUE = logical(1))

  df[,nums] <- round(df[,nums], digits = digits)
  df[,nums][df[,nums] > max_value] <- max_value
  df[,nums][df[,nums] < min_value] <- min_value

  (df)
}
generate_data <- function(N, fractionTraining, fractionValidation, fractionTest, hardness=0.1, noisy = 2) { # hardness from 0 (h) to 1 (hard)
  
  id <- paste0("S#", 1:N)
  absences <- rpois(N, lambda = 5)
  
  A <- matrix(runif(3^2)*2-1, ncol=3) 
  Sigma <- t(A) %*% A
  print(Sigma)
  Sigma[1,] <- Sigma[1,] * c(noisy, hardness, hardness)
  Sigma[2,] <- Sigma[2,] * c(hardness, noisy, hardness)
  Sigma[3,] <- Sigma[3,] * c(hardness, hardness, noisy)
  print(Sigma)
  
  cor_var_means <- c(6.4, 6.7, 7.3)
  print(cor_var_means)
  correlated_vars_df <- as.data.frame(mvrnorm(n = N, mu = cor_var_means, Sigma = Sigma))
  
  correlated_vars_df_cols <- c("prog1_grade", "prog2_grade", "prog_proj_grade")
  colnames(correlated_vars_df) <- correlated_vars_df_cols
  
  correlated_vars_df <- round_df(correlated_vars_df, digits = 1, max_value = 10, min_value = 0)
  
  df <- cbind(id, absences, correlated_vars_df)
  
  sampleSizeTraining   <- floor(fractionTraining   * nrow(df))
  sampleSizeValidation <- floor(fractionValidation * nrow(df))
  sampleSizeTest       <- floor(fractionTest       * nrow(df))

  indicesTraining    <- sort(sample(seq_len(nrow(df)), size=sampleSizeTraining))
  indicesNotTraining <- setdiff(seq_len(nrow(df)), indicesTraining)
  indicesValidation  <- sort(sample(indicesNotTraining, size=sampleSizeValidation))
  indicesTest        <- setdiff(indicesNotTraining, indicesValidation)

  dfTraining   <- df[indicesTraining, ]
  dfValidation <- df[indicesValidation, ]
  dfTest       <- df[indicesTest, ]
  
  return(list(dfTraining, dfValidation, dfTest))
  
}