# Install the package first
#install.packages(c("shiny", "shinydashboard", "ggplot2", "plotly", "DT", "scales"))
#install.packages("BiocManager")
#install.packages("Rcpp")
#install.packages("RcppArmadillo")
#install.packages("Rcpp", lib = .libPaths()[1])
#BiocManager::install("BiocVersion")
#packageVersion("Rcpp")
#packageVersion("RcppArmadillo")

# app.R – Clean version for shinyapps.io deployment
# ─────────────────────────────────────────────────

library(shiny)
library(shinydashboard)
library(ggplot2)
library(plotly)
library(DT)
library(scales)
library(dplyr)
library(forecast)

load("model_objects.RData")

# Pre-compute mean load once (used in UI valueBox)
mean_load <- round(mean(as.numeric(ts_load)), 0)

# ─────────────────────────────────────────────────────────────────────────────
ui <- dashboardPage(
  dashboardHeader(title = "Electricity Load Forecasting"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview",          tabName = "overview",  icon = icon("tachometer-alt")),
      menuItem("Forecasts",         tabName = "forecasts", icon = icon("chart-line")),
      menuItem("Model Accuracy",    tabName = "accuracy",  icon = icon("chart-bar")),
      menuItem("Residuals",         tabName = "residuals", icon = icon("wave-square")),
      menuItem("Predictions Table", tabName = "table",     icon = icon("table"))
    )
  ),
  
  dashboardBody(
    tabItems(
      
      # ── Tab 1: Overview ─────────────────────────────────────────────────────
      tabItem(tabName = "overview",
              fluidRow(
                valueBox(mean_load,   "Mean Load (kW)",      icon = icon("bolt"),        color = "blue"),
                valueBox(1.12,        "Best MASE (RF)",       icon = icon("bullseye"),    color = "green"),
                valueBox(11999,       "Best RMSE (RF) kW",    icon = icon("check"),       color = "green"),
                valueBox("p < 0.001", "SARIMA Ljung-Box",     icon = icon("exclamation"), color = "red")
              ),
              fluidRow(
                box(title = "Full Time Series (2011–2014)", width = 12,
                    status = "primary", solidHeader = TRUE,
                    plotlyOutput("ts_plot", height = 300))
              )
      ),
      
      # ── Tab 2: Forecasts ────────────────────────────────────────────────────
      tabItem(tabName = "forecasts",
              fluidRow(
                box(width = 3,
                    selectInput("model_select", "Show models:",
                                choices  = c("Random Forest", "SARIMA", "ETS",
                                             "Linear Regression", "Seasonal Naive", "Naive"),
                                multiple = TRUE,
                                selected = c("Random Forest", "SARIMA", "Seasonal Naive")),
                    sliderInput("hour_range", "Hour range:",
                                min = 1, max = 168, value = c(1, 168))
                ),
                box(title = "168-Hour Forecast – All Models", width = 9,
                    status = "primary", solidHeader = TRUE,
                    plotlyOutput("forecast_plot", height = 350))
              )
      ),
      
      # ── Tab 3: Model Accuracy ───────────────────────────────────────────────
      tabItem(tabName = "accuracy",
              fluidRow(
                box(title = "Training Set Accuracy", width = 6,
                    status = "info", solidHeader = TRUE,
                    DTOutput("train_table")),
                box(title = "Test Set Accuracy (168-hr horizon)", width = 6,
                    status = "warning", solidHeader = TRUE,
                    DTOutput("test_table"))
              ),
              fluidRow(
                box(title = "MASE Comparison – Test Set", width = 12,
                    status = "primary", solidHeader = TRUE,
                    plotlyOutput("mase_plot", height = 280))
              )
      ),
      
      # ── Tab 4: Residuals ────────────────────────────────────────────────────
      tabItem(tabName = "residuals",
              fluidRow(
                box(width = 3,
                    selectInput("resid_model", "Model:",
                                choices = c("ETS", "SARIMA"), selected = "SARIMA")
                ),
                box(title = "Residual Time Plot", width = 9,
                    status = "primary", solidHeader = TRUE,
                    plotlyOutput("resid_plot", height = 280))
              ),
              fluidRow(
                box(title = "ACF of Residuals", width = 6,
                    status = "info", solidHeader = TRUE,
                    plotOutput("resid_acf", height = 220)),
                box(title = "Residual Histogram", width = 6,
                    status = "info", solidHeader = TRUE,
                    plotlyOutput("resid_hist", height = 220))
              )
      ),
      
      # ── Tab 5: Predictions Table ────────────────────────────────────────────
      tabItem(tabName = "table",
              fluidRow(
                box(title = "Predicted Hourly Load – Next 168 Hours (kW)",
                    width = 12, status = "primary", solidHeader = TRUE,
                    DTOutput("pred_table"))
              )
      )
    )
  )
)

# ─────────────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {
  
  model_colours <- c(
    "Random Forest"     = "#C00000",
    "SARIMA"            = "#FF9900",
    "ETS"               = "#2E75B6",
    "Linear Regression" = "#70AD47",
    "Seasonal Naive"    = "#444444",
    "Naive"             = "#AAAAAA"
  )
  
  # ── Overview: full time series ─────────────────────────────────────────────
  output$ts_plot <- renderPlotly({
    df <- data.frame(
      datetime = seq(as.POSIXct("2011-01-01", tz = "UTC"),
                     by = "hour", length.out = length(ts_load)),
      load = as.numeric(ts_load)
    ) %>%
      mutate(date = as.Date(datetime)) %>%
      group_by(date) %>%
      summarise(load = mean(load), .groups = "drop")
    
    p <- ggplot(df, aes(x = date, y = load)) +
      geom_line(colour = "#2E75B6", linewidth = 0.5) +
      scale_y_continuous(labels = comma) +
      labs(x = NULL, y = "Daily Mean Load (kW)") +
      theme_minimal()
    ggplotly(p) %>% layout(hovermode = "x unified")
  })
  
  # ── Forecasts ──────────────────────────────────────────────────────────────
  output$forecast_plot <- renderPlotly({
    hr1 <- input$hour_range[1]
    hr2 <- input$hour_range[2]
    
    col_map <- c(
      "Random Forest"     = "RF_Forecast",
      "SARIMA"            = "SARIMA_Forecast",
      "ETS"               = "ETS_Forecast",
      "Linear Regression" = "LM_Forecast",
      "Seasonal Naive"    = "SNaive_Forecast",
      "Naive"             = "Naive_Forecast"
    )
    
    df <- forecast_table[hr1:hr2, ]
    p  <- plot_ly()
    for (m in input$model_select) {
      col <- col_map[m]
      p <- add_lines(p, x = df$Hour, y = df[[col]],
                     name = m, line = list(color = model_colours[m]))
    }
    p %>% layout(
      xaxis     = list(title = "Forecast Hour"),
      yaxis     = list(title = "Load (kW)", tickformat = ",.0f"),
      hovermode = "x unified",
      legend    = list(orientation = "h")
    )
  })
  
  # ── Accuracy tables ────────────────────────────────────────────────────────
  output$train_table <- renderDT({
    datatable(comparison_train,
              options = list(pageLength = 6, dom = "t"),
              rownames = FALSE) %>%
      formatRound(columns = c("ME","RMSE","MAE","MAPE","MASE"), digits = 2)
  })
  
  output$test_table <- renderDT({
    datatable(comparison_test,
              options = list(pageLength = 6, dom = "t"),
              rownames = FALSE) %>%
      formatRound(columns = c("ME","RMSE","MAE","MAPE","MASE"), digits = 2)
  })
  
  # ── MASE bar chart ─────────────────────────────────────────────────────────
  output$mase_plot <- renderPlotly({
    df <- comparison_test %>% arrange(MASE)
    plot_ly(df,
            x      = ~reorder(Model, MASE),
            y      = ~MASE,
            type   = "bar",
            marker = list(color = model_colours[df$Model])) %>%
      layout(
        xaxis      = list(title = ""),
        yaxis      = list(title = "MASE (lower is better)"),
        showlegend = FALSE
      )
  })
  
  # ── Residuals ──────────────────────────────────────────────────────────────
  resid_data <- reactive({
    if (input$resid_model == "ETS") residuals(fit_ets)
    else                             residuals(fit_sarima)
  })
  
  output$resid_plot <- renderPlotly({
    r  <- as.numeric(resid_data())
    df <- data.frame(index = seq_along(r), resid = r)
    p  <- ggplot(df, aes(x = index, y = resid)) +
      geom_line(colour = "#2E75B6", linewidth = 0.3) +
      geom_hline(yintercept = 0, colour = "red", linetype = "dashed") +
      labs(x = "Hour", y = "Residual (kW)") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$resid_acf <- renderPlot({
    forecast::Acf(resid_data(), lag.max = 48,
                  main = paste("ACF –", input$resid_model, "residuals"))
  })
  
  output$resid_hist <- renderPlotly({
    r <- as.numeric(resid_data())
    plot_ly(x = r, type = "histogram",
            marker = list(color = "#2E75B6",
                          line  = list(color = "white", width = 0.5))) %>%
      layout(xaxis = list(title = "Residual (kW)"),
             yaxis = list(title = "Count"))
  })
  
  # ── Predictions table ──────────────────────────────────────────────────────
  output$pred_table <- renderDT({
    datatable(forecast_table,
              options  = list(pageLength = 24, scrollY = "320px",
                              scrollCollapse = TRUE),
              rownames = FALSE) %>%
      formatRound(columns = 2:7, digits = 0) %>%
      formatStyle("RF_Forecast",
                  background = "#E1F5EE", fontWeight = "bold")
  })
}

shinyApp(ui = ui, server = server)
