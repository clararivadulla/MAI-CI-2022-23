# This file contains the encodings to find a neural network and train it with Evolutionary Strategies.
# We find the best architecture using Evolution Strategies algorithms
# We train the neural network with Evolution Strategies

iters_test <- 1
iters_arch <- 2
iters_weig <- 1

fit_es <- function(x) {
    # x is a vector of weights used for the network
  model <- nnet(prog_proj_grade / 10 ~ absences + prog1_grade + prog2_grade, wts = x, trace = FALSE,
    data = train, size = size_param, decay = decay_param, lineout = line_param, MaxNWts = 1e6, maxit = 0
  )
  predictions <- predict(model, val)
  rmse <- sqrt(mean((val$prog2_grade - predictions * 10)^2))
  return(rmse)
}

obj.fn.weights <- makeSingleObjectiveFunction(
  name = "Training nnet weights",
  fn = fit_es,
  par.set = makeNumericParamSet("x", len = size_param, lower = -10, upper = 10)
)

get_test_error <- function(weights) {
  res = cmaes(
    obj.fn.weights,
    monitor = makeSimpleMonitor(),
    control = list(
      sigma = 1.5, # initial step size
      lambda = 50, # number of offspring
      stop.ons = c(
        list(stopOnMaxIters(iters_test)), # stop after x iterations
        getDefaultStoppingConditions() # or after default stopping conditions
      )
    )
  )
  model <- nnet(prog_proj_grade / 10 ~ absences + prog1_grade + prog2_grade, wts = res$best.param, trace = FALSE,
    data = train, size = size_param, decay = decay_param, lineout = FALSE, MaxNWts = 1e6, maxit = 0
  )
  predictions <- predict(model, test)
  # since GA maximize the objective function and we want to minimize RMSE
  rmse <- sqrt(mean((test$prog2_grade - predictions * 10)^2))
  return(rmse)
}

fit_nnet_es <- function(x) {
  # set the architecture
  size_param <<- round(x[1])
  decay_param <<- x[2]

  # train the network architecture using ES
  res = cmaes(
    obj.fn.weights,
    monitor = makeSimpleMonitor(),
    control = list(
      sigma = 1.5, # initial step size
      lambda = 50, # number of offspring
      stop.ons = c(
        list(stopOnMaxIters(iters_weig)), # stop after x iterations
        getDefaultStoppingConditions() # or after default stopping conditions
      )
    )
  )

  # return the RMSE score of the network trained with GA
  return(res$best.fitness)
}

obj.fn.arch <- makeSingleObjectiveFunction(
  name = "Training nnet architecture",
  fn = fit_nnet_es,
  par.set = makeParamSet(
    makeNumericParam("s", lower = 1, upper = 128),
    makeNumericParam("d", lower = 1e-5, upper = 0.1)
  )
)

train_es_nnet <- function() {
  tic.clearlog()
  tic()
  res = cmaes(
      obj.fn.arch, 
      monitor = makeSimpleMonitor(),
      control = list(
          sigma = 1.5, # initial step size
          lambda = 50, # number of offspring
          stop.ons = c(
              list(stopOnMaxIters(iters_test)), 
              getDefaultStoppingConditions() 
          )
      )
  )
  toc(log = TRUE, quiet = FALSE)
  log.lst <- tic.log(format = FALSE)
  timings <- unlist(lapply(log.lst, function(x) x$toc - x$tic))
  tic.clearlog()

  results <- data.frame(
    'Type' = 'Evolutionary',
    'Linear' = 0,
    "Size" = round(res$best.param[1]),
    'Decay' = res$best.param[2],
    'Val Error' = res$best.fitness,
    'Test Error' = get_test_error(res$best.param),
    'Time' = sum(timings)
  )

  return(results)
}
