# =============================================================================
# DSCI 725 – Short-Term Electricity Load Forecasting
# Dataset: UCI Electricity Load Diagrams 2011–2014
# Aggregated across all 370 customers → hourly total load
# Models: Naïve/SNaïve, ETS (Holt-Winters), SARIMA, ML (LM + Random Forest)
# =============================================================================

# -----------------------------------------------------------------------------
# 0. INSTALL & LOAD PACKAGES
# -----------------------------------------------------------------------------
packages <- c(
  "tidyverse", "lubridate", "forecast", "tseries",
  "flextable", "randomForest", "caret", "urca"
)
installed <- rownames(installed.packages())
to_install <- packages[!packages %in% installed]
if (length(to_install) > 0)
  install.packages(to_install, dependencies = TRUE)

library(tidyverse)
library(lubridate)
library(forecast)
library(tseries)
library(flextable)
library(randomForest)
library(caret)
library(urca)

# =============================================================================
# PHASE 1: DATA ACQUISITION & PREPARATION
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1  Load raw data
# -----------------------------------------------------------------------------
# Download from: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
# The file is "LD2011_2014.txt" – semicolon-delimited, comma as decimal separator

cat("Loading raw data...\n")
raw <- read.csv(
  "LD2011_2014.txt",
  sep       = ";",
  dec       = ",",
  header    = TRUE,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

# First column is the timestamp
colnames(raw)[1] <- "timestamp"

# -----------------------------------------------------------------------------
# 1.2  Parse timestamps & aggregate to hourly
# -----------------------------------------------------------------------------
cat("Parsing timestamps and aggregating to hourly...\n")

raw$timestamp <- ymd_hms(raw$timestamp)

# Convert all load columns to numeric
load_cols <- setdiff(colnames(raw), "timestamp")
raw[load_cols] <- lapply(raw[load_cols], as.numeric)

# Aggregate all 370 customers → total load per 15-min interval
raw$total_load <- rowSums(raw[load_cols], na.rm = TRUE)

# Aggregate 15-min intervals to hourly means
hourly <- raw %>%
  mutate(hour = floor_date(timestamp, "hour")) %>%
  group_by(hour) %>%
  summarise(load = mean(total_load, na.rm = TRUE), .groups = "drop") %>%
  arrange(hour)

cat(sprintf("Hourly observations: %d\n", nrow(hourly)))

# -----------------------------------------------------------------------------
# 1.3  Handle missing values
# -----------------------------------------------------------------------------
cat("Handling missing values...\n")

# Count NAs
na_count <- sum(is.na(hourly$load))
cat(sprintf("Missing values before fill: %d\n", na_count))

# Forward-fill then back-fill any remaining NAs
hourly$load <- zoo::na.locf(hourly$load, na.rm = FALSE)
hourly$load <- zoo::na.locf(hourly$load, fromLast = TRUE)

cat(sprintf("Missing values after fill: %d\n", sum(is.na(hourly$load))))

# Save cleaned dataset
write.csv(hourly, "hourly_load_cleaned.csv", row.names = FALSE)
cat("Cleaned dataset saved: hourly_load_cleaned.csv\n")

# =============================================================================
# PHASE 2: EXPLORATORY DATA ANALYSIS
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1  Create ts object  (frequency = 24 for daily seasonality)
# -----------------------------------------------------------------------------
# Use the FULL dataset 2011–2014 as described in the project proposal.
# Subsetting to fewer years is not justified when all four years are available
# and consistent — more data improves seasonal parameter estimates for ETS and
# SARIMA and reduces variance in the Random Forest lag-feature matrix.
hourly_sub <- hourly %>% filter(year(hour) %in% 2011:2014)

ts_load <- ts(hourly_sub$load, frequency = 24)  # daily seasonality

cat(sprintf("Time series length: %d hours\n", length(ts_load)))

# -----------------------------------------------------------------------------
# 2.2  Time plot
# -----------------------------------------------------------------------------
png("plot_01_time_series.png", width = 1400, height = 600, res = 120)
plot(ts_load,
     main = "Aggregated Hourly Electricity Load (2011–2014)",
     xlab = "Time (days)", ylab = "Load (kW)",
     col  = "steelblue", lwd = 0.6)
dev.off()
cat("Saved: plot_01_time_series.png\n")

# -----------------------------------------------------------------------------
# 2.3  Summary statistics
# -----------------------------------------------------------------------------
stats_df <- data.frame(
  Statistic = c("Mean", "Median", "Std Dev", "Min", "Max", "Skewness"),
  Value = c(
    round(mean(ts_load), 2),
    round(median(ts_load), 2),
    round(sd(ts_load), 2),
    round(min(ts_load), 2),
    round(max(ts_load), 2),
    round(moments::skewness(ts_load), 4)
  )
)

# Install moments if needed
if (!"moments" %in% installed.packages()) install.packages("moments")
library(moments)
stats_df$Value[6] <- round(skewness(ts_load), 4)
print(stats_df)

# flextable for Word report
ft_stats <- flextable(stats_df) %>%
  set_caption("Table 1. Summary Statistics – Hourly Load") %>%
  autofit()

# -----------------------------------------------------------------------------
# 2.4  Seasonal decomposition (STL – handles multiple seasonality robustly)
# -----------------------------------------------------------------------------
stl_fit <- stl(ts_load, s.window = "periodic", robust = TRUE)

png("plot_02_stl_decomposition.png", width = 1400, height = 900, res = 120)
plot(stl_fit, main = "STL Decomposition of Hourly Load")
dev.off()
cat("Saved: plot_02_stl_decomposition.png\n")

# -----------------------------------------------------------------------------
# 2.5  ACF and PACF
# -----------------------------------------------------------------------------
png("plot_03_acf.png", width = 1200, height = 500, res = 120)
Acf(ts_load, lag.max = 168,
    main = "ACF – Aggregated Hourly Load (lags up to 168 h / 1 week)")
dev.off()

png("plot_04_pacf.png", width = 1200, height = 500, res = 120)
Pacf(ts_load, lag.max = 48,
     main = "PACF – Aggregated Hourly Load (lags up to 48 h)")
dev.off()
cat("Saved: plot_03_acf.png, plot_04_pacf.png\n")

# =============================================================================
# PHASE 3: STATIONARITY DIAGNOSTICS & TRANSFORMATION
# =============================================================================

# -----------------------------------------------------------------------------
# 3.1  Augmented Dickey-Fuller test (raw series)
# -----------------------------------------------------------------------------
cat("\n--- ADF Test: Raw Series ---\n")
adf_raw <- adf.test(ts_load, alternative = "stationary")
print(adf_raw)

adf_table <- data.frame(
  Test      = "ADF – Raw Series",
  Statistic = round(adf_raw$statistic, 4),
  p_value   = round(adf_raw$p.value, 4),
  Result    = ifelse(adf_raw$p.value < 0.05, "Stationary (reject H0)", "Non-stationary")
)

# -----------------------------------------------------------------------------
# 3.2  Log transformation check (Box-Cox lambda)
# -----------------------------------------------------------------------------
lambda <- BoxCox.lambda(ts_load)
cat(sprintf("\nBox-Cox lambda: %.4f\n", lambda))

# lambda = -0.44 → far from 1.0 → transformation is needed
# BoxCox() applies the transformation using the estimated lambda
ts_transformed <- BoxCox(ts_load, lambda = lambda)
cat(sprintf("Box-Cox transformation applied (lambda = %.4f).\n", lambda))

# Verify variance is now more stable
cat(sprintf("Original series SD : %.2f\n", sd(ts_load)))
cat(sprintf("Transformed series SD: %.2f\n", sd(ts_transformed)))
# -----------------------------------------------------------------------------
# 3.3  Differencing check
# -----------------------------------------------------------------------------
# ndiffs gives the number of regular differences needed
n_diffs   <- ndiffs(ts_transformed)
n_sdiffs  <- nsdiffs(ts_transformed)
cat(sprintf("Regular differences needed: %d\n", n_diffs))
cat(sprintf("Seasonal differences needed: %d\n", n_sdiffs))

# ADF after differencing (if needed)
if (n_diffs > 0) {
  ts_diff <- diff(ts_transformed, differences = n_diffs)
  adf_diff <- adf.test(ts_diff, alternative = "stationary")
  adf_table <- rbind(adf_table, data.frame(
    Test      = "ADF – After Differencing",
    Statistic = round(adf_diff$statistic, 4),
    p_value   = round(adf_diff$p.value, 4),
    Result    = ifelse(adf_diff$p.value < 0.05, "Stationary (reject H0)", "Non-stationary")
  ))
  cat("\n--- ADF Test: After Differencing ---\n")
  print(adf_diff)
}

ft_adf <- flextable(adf_table) %>%
  set_caption("Table 2. Augmented Dickey-Fuller Test Results") %>%
  autofit()
print(ft_adf)

# =============================================================================
# PHASE 4: TRAIN / TEST SPLIT
# =============================================================================
# Forecast horizon: 168 hours (1 week) – satisfies rubric requirement
h <- 168

n_total <- length(ts_load)
n_train <- n_total - h

ts_train <- window(ts_load, end   = time(ts_load)[n_train])
ts_test  <- window(ts_load, start = time(ts_load)[n_train + 1])

cat(sprintf("\nTraining obs : %d\nTest obs     : %d\n", length(ts_train), length(ts_test)))

# =============================================================================
# PHASE 5: BASELINE MODELS
# =============================================================================

cat("\n--- Fitting Baseline Models ---\n")

# Naïve
fit_naive  <- naive(ts_train,  h = h)

# Seasonal Naïve (same hour, previous day)
fit_snaive <- snaive(ts_train, h = h)

acc_naive  <- accuracy(fit_naive,  ts_test)
acc_snaive <- accuracy(fit_snaive, ts_test)

cat("Naïve accuracy:\n");  print(acc_naive)
cat("SNaïve accuracy:\n"); print(acc_snaive)

# =============================================================================
# PHASE 6: ETS (HOLT-WINTERS)
# =============================================================================

cat("\n--- Fitting ETS (Holt-Winters) ---\n")

# auto ETS – searches additive vs multiplicative error/trend/seasonality
fit_ets <- ets(ts_train)
cat("ETS model selected:", fit_ets$method, "\n")
summary(fit_ets)

fc_ets   <- forecast(fit_ets, h = h)
acc_ets  <- accuracy(fc_ets, ts_test)
cat("ETS accuracy:\n"); print(acc_ets)

# Residual diagnostics
png("plot_05_ets_residuals.png", width = 1400, height = 700, res = 120)
checkresiduals(fit_ets)
dev.off()
cat("Saved: plot_05_ets_residuals.png\n")

# =============================================================================
# PHASE 7: SARIMA
# =============================================================================

cat("\n--- Fitting SARIMA (auto.arima) ---\n")
# auto.arima with seasonal = TRUE, stepwise = FALSE for thorough search
# NOTE: This can take several minutes on a full multi-year series.
# Reduce stepwise to FALSE only if compute time allows.
fit_sarima <- auto.arima(
  ts_train,
  seasonal    = TRUE,
  stepwise    = TRUE,   # set FALSE for exhaustive search (slower)
  approximation = TRUE,
  trace       = TRUE
)

cat("\nBest SARIMA model:\n")
print(fit_sarima)

fc_sarima  <- forecast(fit_sarima, h = h)
acc_sarima <- accuracy(fc_sarima, ts_test)
cat("SARIMA accuracy:\n"); print(acc_sarima)

# Ljung-Box test on residuals
lb_test <- Box.test(residuals(fit_sarima), lag = 24, type = "Ljung-Box")
cat("\nLjung-Box test (lag=24):\n"); print(lb_test)

# p-value fix: round() collapses 2.2e-16 to 0 — use "< 0.001" instead
format_pvalue <- function(p) {
  if (p < 0.001) "< 0.001" else as.character(round(p, 4))
}

lb_table <- data.frame(
  Test      = "Ljung-Box (lag = 24) – SARIMA residuals",
  Statistic = round(lb_test$statistic, 4),
  df        = as.integer(lb_test$parameter),
  p_value   = format_pvalue(lb_test$p.value),
  Result    = ifelse(lb_test$p.value > 0.05,
                     "No autocorrelation (white noise)",
                     "Autocorrelation detected — model underfit")
)

ft_lb <- flextable(lb_table) %>%
  set_caption("Table 3. Ljung-Box Test – SARIMA Residuals") %>%
  set_header_labels(
    Test      = "Test",
    Statistic = "Q Statistic",
    df        = "df",
    p_value   = "p-value",
    Result    = "Conclusion"
  ) %>%
  bold(part = "header") %>%
  color(i = 1, j = "Result", color = "red") %>%
  align(j = c("Statistic", "df", "p_value"), align = "center", part = "all") %>%
  autofit()
print(ft_lb)

# Residual diagnostics
png("plot_06_sarima_residuals.png", width = 1400, height = 700, res = 120)
checkresiduals(fit_sarima)
dev.off()
cat("Saved: plot_06_sarima_residuals.png\n")

# =============================================================================
# PHASE 8: MACHINE LEARNING MODELS  (Supervised lag-feature approach)
# =============================================================================

cat("\n--- Building ML Feature Matrix ---\n")

# Helper: create lag features from a numeric vector
make_features <- function(x) {
  n <- length(x)
  df <- data.frame(load = x)
  
  # Lag features: 1–24 hours + lag 168 (same hour last week)
  for (lag in c(1:24, 168)) {
    col_name <- paste0("lag_", lag)
    df[[col_name]] <- c(rep(NA, lag), x[1:(n - lag)])
  }
  
  # Cyclical hour-of-day (sin/cos encoding)
  hour_idx <- ((seq_len(n) - 1) %% 24)
  df$sin_hour <- sin(2 * pi * hour_idx / 24)
  df$cos_hour <- cos(2 * pi * hour_idx / 24)
  
  # Day-of-week (1 = Mon … 7 = Sun) – derived from position in series
  # (assumes series starts on 2011-01-01 00:00 which is a Saturday = 6)
  day_idx <- ((seq_len(n) - 1) %/% 24 + 5) %% 7   # 0=Mon … 6=Sun
  for (d in 0:6) {
    df[[paste0("dow_", d)]] <- as.integer(day_idx == d)
  }
  
  df
}

full_vec   <- as.numeric(ts_load)
feature_df <- make_features(full_vec)

# Drop rows with NA (first 168 rows due to lag_168)
feature_df <- feature_df %>% drop_na()

# Align indices: feature_df starts at row 169 of ts_load
offset <- nrow(feature_df) - nrow(drop_na(make_features(full_vec))) + 169 - 1
# Simpler approach: the first valid row in the original is row 169
first_valid <- 169
n_feat      <- nrow(feature_df)

# Train/test split matching the ts split
# train: rows up to (n_train - first_valid + 1) in feature_df
train_end_feat <- n_train - first_valid + 1
if (train_end_feat <= 0 || train_end_feat >= n_feat) {
  stop("Train/test split incompatible with lag window. Reduce h or use more data.")
}

train_feat <- feature_df[1:train_end_feat, ]
test_feat  <- feature_df[(train_end_feat + 1):min(train_end_feat + h, n_feat), ]

X_train <- train_feat %>% select(-load)
y_train <- train_feat$load
X_test  <- test_feat  %>% select(-load)
y_test  <- test_feat$load

cat(sprintf("ML train rows: %d | ML test rows: %d\n", nrow(X_train), nrow(X_test)))

# ------------------------------------------------------------------
# 8.1  Linear Regression
# ------------------------------------------------------------------
cat("\n--- Fitting Linear Regression ---\n")
fit_lm <- lm(load ~ ., data = train_feat)

# ── In-sample (training) predictions
pred_lm_train <- predict(fit_lm, newdata = X_train)

# ── Out-of-sample (test) predictions
pred_lm_test  <- predict(fit_lm, newdata = X_test)

# MASE denominator: mean |error| of seasonal naïve (lag-24) on training set
naive_errors <- abs(diff(y_train, lag = 24))
mase_denom   <- mean(naive_errors)

# Training metrics
me_lm_train   <- mean(pred_lm_train - y_train)
mae_lm_train  <- mean(abs(pred_lm_train - y_train))
rmse_lm_train <- sqrt(mean((pred_lm_train - y_train)^2))
mape_lm_train <- mean(abs((pred_lm_train - y_train) / y_train)) * 100
mase_lm_train <- mae_lm_train / mase_denom

# Test metrics
me_lm_test   <- mean(pred_lm_test - y_test)
mae_lm_test  <- mean(abs(pred_lm_test - y_test))
rmse_lm_test <- sqrt(mean((pred_lm_test - y_test)^2))
mape_lm_test <- mean(abs((pred_lm_test - y_test) / y_test)) * 100
mase_lm_test <- mae_lm_test / mase_denom

cat(sprintf("LM Train – ME: %.2f | MAE: %.2f | RMSE: %.2f | MASE: %.4f | MAPE: %.2f%%\n",
            me_lm_train, mae_lm_train, rmse_lm_train, mase_lm_train, mape_lm_train))
cat(sprintf("LM Test  – ME: %.2f | MAE: %.2f | RMSE: %.2f | MASE: %.4f | MAPE: %.2f%%\n",
            me_lm_test,  mae_lm_test,  rmse_lm_test,  mase_lm_test,  mape_lm_test))

# ------------------------------------------------------------------
# 8.2  Random Forest
# ------------------------------------------------------------------
cat("\n--- Fitting Random Forest (ntree=200) ---\n")
set.seed(42)
fit_rf <- randomForest(
  x          = X_train,
  y          = y_train,
  ntree      = 200,
  mtry       = floor(sqrt(ncol(X_train))),
  importance = TRUE
)

# ── In-sample (training) predictions  — RF reports OOB error internally;
#    we also compute fitted values on the training set for the accuracy table.
pred_rf_train <- predict(fit_rf, newdata = X_train)

# ── Out-of-sample (test) predictions
pred_rf_test  <- predict(fit_rf, newdata = X_test)

# Training metrics
me_rf_train   <- mean(pred_rf_train - y_train)
mae_rf_train  <- mean(abs(pred_rf_train - y_train))
rmse_rf_train <- sqrt(mean((pred_rf_train - y_train)^2))
mape_rf_train <- mean(abs((pred_rf_train - y_train) / y_train)) * 100
mase_rf_train <- mae_rf_train / mase_denom

# Test metrics
me_rf_test   <- mean(pred_rf_test - y_test)
mae_rf_test  <- mean(abs(pred_rf_test - y_test))
rmse_rf_test <- sqrt(mean((pred_rf_test - y_test)^2))
mape_rf_test <- mean(abs((pred_rf_test - y_test) / y_test)) * 100
mase_rf_test <- mae_rf_test / mase_denom

cat(sprintf("RF Train – ME: %.2f | MAE: %.2f | RMSE: %.2f | MASE: %.4f | MAPE: %.2f%%\n",
            me_rf_train, mae_rf_train, rmse_rf_train, mase_rf_train, mape_rf_train))
cat(sprintf("RF Test  – ME: %.2f | MAE: %.2f | RMSE: %.2f | MASE: %.4f | MAPE: %.2f%%\n",
            me_rf_test,  mae_rf_test,  rmse_rf_test,  mase_rf_test,  mape_rf_test))

# Variable importance plot
png("plot_07_rf_importance.png", width = 1000, height = 700, res = 120)
varImpPlot(fit_rf, main = "Random Forest – Variable Importance")
dev.off()
cat("Saved: plot_07_rf_importance.png\n")

# =============================================================================
# PHASE 9: MODEL COMPARISON TABLE  (Training AND Test — rubric requirement)
# =============================================================================

cat("\n--- Building Accuracy Comparison Tables (Train + Test) ---\n")

# Helper: extract one row from accuracy() output for classical models
get_acc <- function(acc_mat, set) {
  row <- acc_mat[set, ]
  data.frame(
    ME   = round(row["ME"],   2),
    RMSE = round(row["RMSE"], 2),
    MAE  = round(row["MAE"],  2),
    MAPE = round(row["MAPE"], 2),
    MASE = round(row["MASE"], 4)
  )
}

# ── 9.1  TRAINING SET accuracy table ──────────────────────────────────────────
comparison_train <- data.frame(
  Model = c("Naïve", "Seasonal Naïve", "ETS", "SARIMA",
            "Linear Regression", "Random Forest"),
  Set = "Training",
  rbind(
    get_acc(acc_naive,  "Training set"),
    get_acc(acc_snaive, "Training set"),
    get_acc(acc_ets,    "Training set"),
    get_acc(acc_sarima, "Training set"),
    data.frame(ME   = round(me_lm_train,   2),
               RMSE = round(rmse_lm_train, 2),
               MAE  = round(mae_lm_train,  2),
               MAPE = round(mape_lm_train, 2),
               MASE = round(mase_lm_train, 4)),
    data.frame(ME   = round(me_rf_train,   2),
               RMSE = round(rmse_rf_train, 2),
               MAE  = round(mae_rf_train,  2),
               MAPE = round(mape_rf_train, 2),
               MASE = round(mase_rf_train, 4))
  )
)

cat("\n=== TRAINING SET ACCURACY ===\n")
print(comparison_train)

ft_train <- flextable(comparison_train) %>%
  set_caption("Table 4a. Forecast Accuracy – Training Set (All Models)") %>%
  bold(part = "header") %>%
  color(i = which.min(comparison_train$MASE), color = "darkgreen") %>%
  autofit()
print(ft_train)

# ── 9.2  TEST SET accuracy table ──────────────────────────────────────────────
comparison_test <- data.frame(
  Model = c("Naïve", "Seasonal Naïve", "ETS", "SARIMA",
            "Linear Regression", "Random Forest"),
  Set = "Test",
  rbind(
    get_acc(acc_naive,  "Test set"),
    get_acc(acc_snaive, "Test set"),
    get_acc(acc_ets,    "Test set"),
    get_acc(acc_sarima, "Test set"),
    data.frame(ME   = round(me_lm_test,   2),
               RMSE = round(rmse_lm_test, 2),
               MAE  = round(mae_lm_test,  2),
               MAPE = round(mape_lm_test, 2),
               MASE = round(mase_lm_test, 4)),
    data.frame(ME   = round(me_rf_test,   2),
               RMSE = round(rmse_rf_test, 2),
               MAE  = round(mae_rf_test,  2),
               MAPE = round(mape_rf_test, 2),
               MASE = round(mase_rf_test, 4))
  )
)

cat("\n=== TEST SET ACCURACY ===\n")
print(comparison_test)

ft_test <- flextable(comparison_test) %>%
  set_caption("Table 4b. Forecast Accuracy – Test Set / 168-Hour Horizon (All Models)") %>%
  bold(part = "header") %>%
  color(i = which.min(comparison_test$MASE), color = "darkgreen") %>%
  autofit()
print(ft_test)

# ── 9.3  Combined long-format table (optional — useful for a single Word table)
comparison_combined <- rbind(comparison_train, comparison_test)

ft_combined <- flextable(comparison_combined) %>%
  set_caption("Table 4. Forecast Accuracy – All Models, Training and Test Sets") %>%
  bold(part = "header") %>%
  merge_v(j = "Model") %>%          # merge repeated model name cells vertically
  theme_vanilla() %>%
  autofit()
print(ft_combined)

# Best model on test set based on MASE
best_model <- comparison_test$Model[which.min(comparison_test$MASE)]
cat(sprintf("\nBest model (test MASE): %s\n", best_model))

# ── 9.4  Save accuracy tables to CSV for reference
write.csv(comparison_train,    "accuracy_training.csv",    row.names = FALSE)
write.csv(comparison_test,     "accuracy_test.csv",        row.names = FALSE)
write.csv(comparison_combined, "accuracy_combined.csv",    row.names = FALSE)
cat("Saved: accuracy_training.csv, accuracy_test.csv, accuracy_combined.csv\n")

# =============================================================================
# PHASE 10: FINAL FORECAST – NEXT 168 HOURS (1 WEEK) – ALL 6 MODELS
# =============================================================================

cat("\n--- Generating Final Forecasts (all 6 models) ---\n")

# -----------------------------------------------------------------------------
# 10.1  Re-fit classical models on the FULL series
# -----------------------------------------------------------------------------
fit_ets_full    <- ets(ts_load)
fit_sarima_full <- auto.arima(ts_load, seasonal = TRUE, stepwise = TRUE,
                              approximation = TRUE)

fc_ets_final    <- forecast(fit_ets_full,    h = h)
fc_sarima_final <- forecast(fit_sarima_full, h = h)

# -----------------------------------------------------------------------------
# 10.2  Naïve and Seasonal Naïve on full series
# -----------------------------------------------------------------------------
fc_naive_final  <- naive(ts_load,  h = h)
fc_snaive_final <- snaive(ts_load, h = h)

# -----------------------------------------------------------------------------
# 10.3  ML forecasts for next 168 hours (direct multi-step strategy)
#
#  For each future hour t+k (k = 1 … 168) we build one feature row using:
#    - Lags 1–24 and lag 168 drawn from the known history + already-predicted values
#    - The same sin/cos hour and day-of-week encodings as during training
#
#  We use the RECURSIVE strategy: each new prediction is appended to the
#  history so it can serve as a lagged input for subsequent steps.
# -----------------------------------------------------------------------------
cat("  Building ML recursive 168-step forecast...\n")

# Start from the full observed series
extended <- as.numeric(ts_load)          # length = n_total
n_ext    <- length(extended)

# Hour-of-day index for the NEXT h hours (continues from last observed hour)
# The series starts at 2011-01-01 00:00 (hour index 0).
# Last observed hour index = n_total - 1 (0-based).
last_hour_idx <- (n_ext - 1) %% 24      # 0-based hour within day
last_day_idx  <- ((n_ext - 1) %/% 24 + 5) %% 7  # 0-based day-of-week (0=Mon)

pred_lm_final <- numeric(h)
pred_rf_final <- numeric(h)

for (k in seq_len(h)) {
  
  # ---- feature values for step k ----
  # lag_1 … lag_24: look back into extended (already includes prior predictions)
  lag_vals <- sapply(c(1:24, 168), function(lag) {
    idx <- length(extended) - lag + 1     # 1-based index into extended
    if (idx < 1) NA_real_ else extended[idx]
  })
  names(lag_vals) <- paste0("lag_", c(1:24, 168))
  
  # Cyclical hour encoding
  future_hour_idx <- (last_hour_idx + k) %% 24
  sin_hour <- sin(2 * pi * future_hour_idx / 24)
  cos_hour <- cos(2 * pi * future_hour_idx / 24)
  
  # Day-of-week dummies
  future_day_idx <- (last_day_idx + (last_hour_idx + k) %/% 24) %% 7
  dow_dummies <- setNames(
    as.integer(0:6 == future_day_idx),
    paste0("dow_", 0:6)
  )
  
  # Assemble feature row — must match column order of X_train exactly
  feat_row <- as.data.frame(t(c(lag_vals, sin_hour = sin_hour,
                                cos_hour = cos_hour, dow_dummies)))
  
  # Ensure column types match training data
  feat_row[] <- lapply(feat_row, as.numeric)
  
  # ---- predictions ----
  p_lm <- predict(fit_lm, newdata = feat_row)
  p_rf <- predict(fit_rf, newdata = feat_row)
  
  pred_lm_final[k] <- p_lm
  pred_rf_final[k] <- p_rf
  
  # Append the average of LM and RF as the "observed" value for recursive lags
  # (you can also use just one model; RF is typically more accurate)
  extended <- c(extended, p_rf)
}

cat("  ML recursive forecast complete.\n")

# -----------------------------------------------------------------------------
# 10.4  Assemble the complete 6-model forecast table
# -----------------------------------------------------------------------------
forecast_table <- data.frame(
  Hour            = 1:h,
  Naive_Forecast  = round(as.numeric(fc_naive_final$mean),  2),
  SNaive_Forecast = round(as.numeric(fc_snaive_final$mean), 2),
  ETS_Forecast    = round(as.numeric(fc_ets_final$mean),    2),
  SARIMA_Forecast = round(as.numeric(fc_sarima_final$mean), 2),
  LM_Forecast     = round(pred_lm_final,                    2),
  RF_Forecast     = round(pred_rf_final,                    2)
)

# Print summary (first and last 6 rows) to console
cat("\nForecast table – first 6 rows:\n")
print(head(forecast_table, 6))
cat("\nForecast table – last 6 rows:\n")
print(tail(forecast_table, 6))

# Flextable for Word report
ft_forecast <- flextable(forecast_table) %>%
  set_caption(
    paste0("Table 5. Predicted Hourly Load – Next ", h,
           " Hours (1 Week Ahead) – All Models (kW)")
  ) %>%
  set_header_labels(
    Hour            = "Hour",
    Naive_Forecast  = "Naïve (kW)",
    SNaive_Forecast = "SNaïve (kW)",
    ETS_Forecast    = "ETS (kW)",
    SARIMA_Forecast = "SARIMA (kW)",
    LM_Forecast     = "LM (kW)",
    RF_Forecast     = "RF (kW)"
  ) %>%
  colformat_num(j = 2:7, digits = 0) %>%   # whole numbers for readability
  bold(part = "header") %>%
  bg(i = seq(2, h, 2), bg = "#F2F2F2") %>% # alternating row shading
  autofit()

print(ft_forecast)

# Save to CSV for copy-paste into Word
write.csv(forecast_table, "forecast_next_168h_all_models.csv", row.names = FALSE)
cat("Saved: forecast_next_168h_all_models.csv\n")

# Quick sanity check – plot all 6 forecasts
png("plot_09_all_model_forecasts.png", width = 1600, height = 700, res = 120)
par(mar = c(5, 5, 4, 8))
matplot(
  forecast_table$Hour,
  forecast_table[, -1],
  type = "l", lty = 1,
  col  = c("gray60", "gray30", "dodgerblue", "darkorange", "forestgreen", "firebrick"),
  lwd  = c(1, 1, 2, 2, 1.5, 1.5),
  xlab = "Forecast Hour", ylab = "Load (kW)",
  main = "168-Hour Ahead Forecast – All 6 Models"
)
legend("topright", inset = c(-0.12, 0),
       legend = c("Naïve", "SNaïve", "ETS", "SARIMA", "LM", "RF"),
       col    = c("gray60", "gray30", "dodgerblue", "darkorange", "forestgreen", "firebrick"),
       lty = 1, lwd = c(1, 1, 2, 2, 1.5, 1.5),
       cex = 0.8, bty = "n", xpd = TRUE)
dev.off()
cat("Saved: plot_09_all_model_forecasts.png\n")

# =============================================================================
# PHASE 11: COMBINED VISUALISATION
# (time series + fitted values + forecast, all model families in one plot)
# =============================================================================

cat("\n--- Plotting Combined Forecast Chart ---\n")

# Use last 30 days of training data for readability
plot_start <- max(1, n_train - 30 * 24)
ts_plot    <- window(ts_load, start = time(ts_load)[plot_start])

png("plot_08_combined_forecast.png", width = 1600, height = 700, res = 120)
par(mar = c(5, 5, 4, 2))

# Actual series (last 30 days + test week)
plot(ts_plot,
     col  = "black", lwd = 1.2,
     xlim = c(time(ts_load)[plot_start],
              time(ts_load)[n_total] + h / 24),
     ylim = range(ts_load, fc_ets_final$upper, na.rm = TRUE),
     main = "Hourly Load: Actual, Fitted & Forecast (All Model Families)",
     xlab = "Time (days)", ylab = "Load (kW)")

# Fitted values – ETS (in-sample, training only)
lines(fitted(fit_ets),    col = "dodgerblue",  lwd = 1, lty = 2)

# Fitted values – SARIMA
lines(fitted(fit_sarima), col = "darkorange",  lwd = 1, lty = 2)

# ETS forecast
lines(fc_ets_final$mean,    col = "dodgerblue", lwd = 2)

# SARIMA forecast
lines(fc_sarima_final$mean, col = "darkorange", lwd = 2)

# Naïve forecast (test period) — already computed as fc_snaive_final
lines(fc_snaive_final$mean,
      col = "gray50", lwd = 1.5, lty = 3)

legend("topleft",
       legend = c("Actual", "ETS fitted", "SARIMA fitted",
                  "ETS forecast", "SARIMA forecast", "SNaïve forecast"),
       col    = c("black", "dodgerblue", "darkorange",
                  "dodgerblue", "darkorange", "gray50"),
       lty    = c(1, 2, 2, 1, 1, 3),
       lwd    = c(1.2, 1, 1, 2, 2, 1.5),
       cex    = 0.8, bty = "n")

dev.off()
cat("Saved: plot_08_combined_forecast.png\n")

# =============================================================================
# PHASE 12: ROLLING-ORIGIN CROSS-VALIDATION (MASE evaluation)
# =============================================================================

cat("\n--- Rolling-Origin Cross-Validation ---\n")

# Use a smaller window for speed; adjust init_window and n_rolls as needed
init_window <- 24 * 30   # 30 days initial training
step_size   <- 24        # advance by 1 day each roll
n_rolls     <- 14        # 14 rolls = 2-week evaluation

cv_results <- data.frame(
  Roll = integer(), Model = character(),
  MAE = numeric(), RMSE = numeric(), MASE = numeric()
)

for (i in seq_len(n_rolls)) {
  end_idx   <- init_window + (i - 1) * step_size
  start_idx <- 1
  if (end_idx + h > n_total) break
  
  tr <- ts(as.numeric(ts_load)[start_idx:end_idx],   frequency = 24)
  te <- as.numeric(ts_load)[(end_idx + 1):(end_idx + h)]
  denom <- mean(abs(diff(as.numeric(tr), lag = 24)))
  
  # Naïve
  pn  <- as.numeric(snaive(tr, h = h)$mean)
  cv_results <- rbind(cv_results, data.frame(
    Roll = i, Model = "SNaive",
    MAE  = mean(abs(pn - te)),
    RMSE = sqrt(mean((pn - te)^2)),
    MASE = mean(abs(pn - te)) / denom
  ))
  
  # ETS
  pe  <- as.numeric(forecast(ets(tr), h = h)$mean)
  cv_results <- rbind(cv_results, data.frame(
    Roll = i, Model = "ETS",
    MAE  = mean(abs(pe - te)),
    RMSE = sqrt(mean((pe - te)^2)),
    MASE = mean(abs(pe - te)) / denom
  ))
  
  # SARIMA
  ps  <- tryCatch({
    as.numeric(forecast(auto.arima(tr, seasonal = TRUE,
                                   stepwise = TRUE,
                                   approximation = TRUE), h = h)$mean)
  }, error = function(e) rep(NA, h))
  if (!all(is.na(ps))) {
    cv_results <- rbind(cv_results, data.frame(
      Roll = i, Model = "SARIMA",
      MAE  = mean(abs(ps - te), na.rm = TRUE),
      RMSE = sqrt(mean((ps - te)^2, na.rm = TRUE)),
      MASE = mean(abs(ps - te), na.rm = TRUE) / denom
    ))
  }
  
  cat(sprintf("Roll %2d / %d complete\n", i, n_rolls))
}

# Summarise CV results
cv_summary <- cv_results %>%
  group_by(Model) %>%
  summarise(
    Mean_MAE  = round(mean(MAE,  na.rm = TRUE), 2),
    Mean_RMSE = round(mean(RMSE, na.rm = TRUE), 2),
    Mean_MASE = round(mean(MASE, na.rm = TRUE), 4),
    .groups = "drop"
  ) %>%
  arrange(Mean_MASE)

print(cv_summary)

ft_cv <- flextable(cv_summary) %>%
  set_caption("Table 6. Rolling-Origin Cross-Validation – Average Accuracy") %>%
  bold(part = "header") %>%
  autofit()
print(ft_cv)

##Save file
save(ts_load, forecast_table, comparison_train, comparison_test,
      fit_ets, fit_sarima, fit_lm, fit_rf,
      fc_ets_final, fc_sarima_final, fc_naive_final, fc_snaive_final,
     pred_lm_final, pred_rf_final, hourly_sub,
     file = "model_objects.RData")
# =============================================================================
# DONE
# =============================================================================

cat("\n=== All phases complete. Output files generated: ===\n")
cat("  hourly_load_cleaned.csv\n")
cat("  forecast_next_168h_all_models.csv    ← 6-model 168-hour forecast table\n")
cat("  plot_01_time_series.png\n")
cat("  plot_02_stl_decomposition.png\n")
cat("  plot_03_acf.png\n")
cat("  plot_04_pacf.png\n")
cat("  plot_05_ets_residuals.png\n")
cat("  plot_06_sarima_residuals.png\n")
cat("  plot_07_rf_importance.png\n")
cat("  plot_08_combined_forecast.png\n")
cat("  plot_09_all_model_forecasts.png      ← all 6 models over 168-hour horizon\n")
cat("\nFlextable objects ready for Word export:\n")
cat("  ft_stats, ft_adf, ft_lb, ft_comparison, ft_forecast, ft_cv\n")