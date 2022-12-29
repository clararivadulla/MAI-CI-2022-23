# This file contains the encodings to find a neural network using backpropagation.
# We find the best architecture using genetic algorithms
# We train the neural network with backpropagation

iters_test <- 500
iters_arch <- 50
iters_weig <- 5

linear <- 0:1                     # use a linear out or not
b0 <- decimal2binary(max(linear)) # max number of bits requires
l0 <- length(b0)                  # length of bits needed for encoding

sizes <- 0:3                      # range of values to search for the number of nodes in the hidden layer
b1 <- decimal2binary(max(sizes))  # max number of bits requires
l1 <- length(b1)                  # length of bits needed for encoding

decays  <- 0:3                    # range of values to search for the the decay
b2 <- decimal2binary(max(decays)) # max number of bits requires
l2 <- length(b2)                  # length of bits needed for encoding

decoder <- function(x) {
  x <- gray2binary(x) # we use gray2binary to encode the binary such that each different value only changes in one bit
  lin <- binary2decimal(x[l0])
  siz <- 2**binary2decimal(x[l0 + 1:(l1 + l0)])
  dec <- 10**(-binary2decimal(x[(l0 + l1 + 1):(l0 + l1 + l2)]) - 1)
  out <- structure(c(lin, siz, dec), names = c("linear", "size", "decay"))
  return(out)
}

fit_nnet_architecture <- function(x) {
  pair <- decoder(x)
  line <- pair[1]
  size <- pair[2]
  decay <- pair[3]

  model <- nnet(prog_proj_grade / 10 ~ absences + prog1_grade + prog2_grade, trace = FALSE,
    data = train, size = size, decay = decay, lineout = line, MaxNWts = 1e6, maxit = iters_weig
  )
  predictions <- predict(model, val)
  # since GA maximize the objective function and we want to minimize RMSE
  rmse <- sqrt(mean((val$prog2_grade - predictions * 10)^2))

  return(-rmse)
}

get_test_error = function(params) {
  model <- nnet(prog_proj_grade / 10 ~ absences + prog1_grade + prog2_grade, trace = FALSE,
    data = train, size = params[2], decay = params[3], lineout = params[1], MaxNWts = 1e6, maxit = iters_test
  ) 
  predictions <- predict(model, test)
  rmse <- sqrt(mean((test$prog2_grade - predictions * 10)^2))
  return(rmse)
}

train_nnet <- function() {
  tic.clearlog()
  tic()
  GA = ga(
      type = "binary", fitness = fit_nnet_architecture, nBits = l0 + l1 + l2,
      maxiter = iters_arch, popSize = 50, seed = seed, keepBest = TRUE, monitor = FALSE
  )
  toc(log = TRUE, quiet = FALSE)
  log.lst <- tic.log(format = FALSE)
  timings <- unlist(lapply(log.lst, function(x) x$toc - x$tic))
  tic.clearlog()

  values <- decoder(GA@solution)

  results <- data.frame(
    'Type' = 'Backpropagation',
    'Linear' = values[1],
    "Size" = values[2],
    'Decay' = values[3],
    'Val Error' = -GA@fitnessValue,
    'Test Error' = get_test_error(decoder(GA@solution)),
    'Time' = sum(timings)
  )
  return(list(GA, results))
  
}
