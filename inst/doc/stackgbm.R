## ----include=FALSE------------------------------------------------------------
knitr::opts_chunk$set(
  comment = "#>",
  collapse = TRUE
)

run <- if (rlang::is_installed(c("catboost", "lightgbm", "xgboost"))) TRUE else FALSE
knitr::opts_chunk$set(eval = run)

## ----message=FALSE------------------------------------------------------------
library("stackgbm")

## -----------------------------------------------------------------------------
sim_data <- msaenet::msaenet.sim.binomial(
  n = 1000,
  p = 50,
  rho = 0.6,
  coef = rnorm(25, mean = 0, sd = 10),
  snr = 1,
  p.train = 0.8,
  seed = 42
)

x_train <- sim_data$x.tr
x_test <- sim_data$x.te
y_train <- as.vector(sim_data$y.tr)
y_test <- as.vector(sim_data$y.te)

## ----eval=FALSE---------------------------------------------------------------
#  params_xgboost <- cv_xgboost(x_train, y_train)
#  params_lightgbm <- cv_lightgbm(x_train, y_train)
#  params_catboost <- cv_catboost(x_train, y_train)

## ----eval=FALSE, echo=FALSE---------------------------------------------------
#  saveRDS(params_xgboost, file = "vignettes/params_xgboost.rds")
#  saveRDS(params_lightgbm, file = "vignettes/params_lightgbm.rds")
#  saveRDS(params_catboost, file = "vignettes/params_catboost.rds")
#  
#  temp_dir <- "catboost_info"
#  temp_file <- "lightgbm.model"
#  if (dir.exists(temp_dir)) unlink(temp_dir, recursive = TRUE)
#  if (file.exists(temp_file)) unlink(temp_file)

## ----echo=FALSE---------------------------------------------------------------
params_xgboost <- readRDS("params_xgboost.rds")
params_lightgbm <- readRDS("params_lightgbm.rds")
params_catboost <- readRDS("params_catboost.rds")

## -----------------------------------------------------------------------------
model_stackgbm <- stackgbm(
  sim_data$x.tr,
  sim_data$y.tr,
  params = list(
    params_xgboost,
    params_lightgbm,
    params_catboost
  )
)

## -----------------------------------------------------------------------------
roc_stackgbm_train <- pROC::roc(
  y_train,
  predict(model_stackgbm, x_train)$prob,
  quiet = TRUE
)
roc_stackgbm_test <- pROC::roc(
  y_test,
  predict(model_stackgbm, x_test)$prob,
  quiet = TRUE
)
roc_stackgbm_train$auc
roc_stackgbm_test$auc

## ----message=FALSE------------------------------------------------------------
model_xgboost <- xgboost_train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = params_xgboost$max_depth,
    eta = params_xgboost$eta
  ),
  data = xgboost_dmatrix(x_train, label = y_train),
  nrounds = params_xgboost$nrounds
)

model_lightgbm <- lightgbm_train(
  data = x_train,
  label = y_train,
  params = list(
    objective = "binary",
    learning_rate = params_lightgbm$learning_rate,
    num_iterations = params_lightgbm$num_iterations,
    max_depth = params_lightgbm$max_depth,
    num_leaves = 2^params_lightgbm$max_depth - 1
  ),
  verbose = -1
)

model_catboost <- catboost_train(
  catboost_load_pool(data = x_train, label = y_train),
  NULL,
  params = list(
    loss_function = "Logloss",
    iterations = params_catboost$iterations,
    depth = params_catboost$depth,
    logging_level = "Silent"
  )
)

## -----------------------------------------------------------------------------
roc_xgboost_train <- pROC::roc(
  y_train,
  predict(model_xgboost, x_train),
  quiet = TRUE
)
roc_xgboost_test <- pROC::roc(
  y_test,
  predict(model_xgboost, x_test),
  quiet = TRUE
)
roc_xgboost_train$auc
roc_xgboost_test$auc

## -----------------------------------------------------------------------------
roc_lightgbm_train <- pROC::roc(
  y_train,
  predict(model_lightgbm, x_train),
  quiet = TRUE
)
roc_lightgbm_test <- pROC::roc(
  y_test,
  predict(model_lightgbm, x_test),
  quiet = TRUE
)
roc_lightgbm_train$auc
roc_lightgbm_test$auc

## -----------------------------------------------------------------------------
roc_catboost_train <- pROC::roc(
  y_train,
  catboost_predict(
    model_catboost,
    catboost_load_pool(data = x_train, label = NULL)
  ),
  quiet = TRUE
)
roc_catboost_test <- pROC::roc(
  y_test,
  catboost_predict(
    model_catboost,
    catboost_load_pool(data = x_test, label = NULL)
  ),
  quiet = TRUE
)
roc_catboost_train$auc
roc_catboost_test$auc

## ----echo=FALSE---------------------------------------------------------------
df <- as.data.frame(matrix(NA, ncol = 4, nrow = 2))
names(df) <- c("stackgbm", "xgboost", "lightgbm", "catboost")
rownames(df) <- c("Training", "Testing")

df$stackgbm <- c(roc_stackgbm_train$auc, roc_stackgbm_test$auc)
df$xgboost <- c(roc_xgboost_train$auc, roc_xgboost_test$auc)
df$lightgbm <- c(roc_lightgbm_train$auc, roc_lightgbm_test$auc)
df$catboost <- c(roc_catboost_train$auc, roc_catboost_test$auc)

knitr::kable(
  df,
  digits = 4,
  caption = "AUC values from four models on training and testing set"
)

## -----------------------------------------------------------------------------
pal <- c("#e15759", "#f28e2c", "#59a14f", "#4e79a7", "#76b7b2")

plot(pROC::smooth(roc_stackgbm_test), col = pal[1], lwd = 1)
plot(pROC::smooth(roc_xgboost_test), col = pal[2], lwd = 1, add = TRUE)
plot(pROC::smooth(roc_lightgbm_test), col = pal[3], lwd = 1, add = TRUE)
plot(pROC::smooth(roc_catboost_test), col = pal[4], lwd = 1, add = TRUE)
legend(
  "bottomright",
  col = pal,
  lwd = 2,
  legend = c("stackgbm", "xgboost", "lightgbm", "catboost")
)

