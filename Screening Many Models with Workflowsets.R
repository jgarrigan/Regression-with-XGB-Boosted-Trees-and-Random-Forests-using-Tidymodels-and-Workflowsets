pacman::p_load(
  tidymodels,
  tidyverse,
  doParallel,
  janitor,
  AmesHousing,
  vip,
  randomForest,
  leaflet,
  sf,
  RColorBrewer,
  ggspatial,
  ggforce,
  baguette,
  bonsai,
  finetune,
  lightgbm,
  GGally,
  ggcorrplot,
  reshape2,
  ggpmisc
)

pacman::p_install_gh("brunocarlin/tidy.outliers")

set.seed(1234)

# load the housing data and clean names
ames_data <- make_ames() %>%
  janitor::clean_names()


# EDA ---------------------------------------------------------------------

# PLOT THE DISTRIBUTION OF SALE PRICE FOR THE AMES HOUSING DATA
ggplot(ames_data, aes(x = sale_price)) +
  geom_histogram(bins = 50, col = "white") +
  scale_x_continuous(labels = scales::dollar_format()) +
  labs(
    title = "Sale price of houses in Ames, Iowa",
    x = "Sale Price",
    y = "Count"
  ) +
  theme_minimal()

# PLOT THE LOG SCALED VERSION OF SALE PRICE
ggplot(ames_data, aes(x = sale_price)) +
  geom_histogram(bins = 50, col = "white") +
  scale_x_log10(labels = scales::dollar_format()) +
  labs(
    title = "Sale prices of houses in Ames, Iowa after a log (base 10) transformation",
    x = "Sale Price",
    y = "Count"
  ) +
  theme_minimal()

ames_corr <- ames_data %>%
  select(where(is.numeric)) %>%
  cor()

ames_long <- melt(ames_corr)

ggplot(ames_long, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = muted("red"), high = muted("blue"), mid = "white", midpoint = 0, limit = c(-1, 1), space = "Lab", name = "Pearson\nCorrelation") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_blank(),
    axis.title.y = element_blank()
  ) +
  coord_fixed()

# Generate a color palette that ranges from blue to red
pal <- colorNumeric(palette = "YlOrRd", domain = ames_data$sale_price)

# ARE THERE SPECIFIC REGIONS OF AMES IOWA WITH HIGHER HOUSE PRICES
leaflet(data = ames_data) %>%
  addTiles() %>% # Add default OpenStreetMap map tiles
  addCircles(
    ~longitude, ~latitude,
    color = ~ pal(sale_price),
    fillColor = ~ pal(sale_price),
    fillOpacity = 0.7,
    radius = 5, # You can adjust the radius size
    popup = ~ paste0("Sale Price: $", sale_price)
  ) %>%
  addLegend(
    "bottomright",
    pal = pal,
    values = ~sale_price,
    title = "Sale Price",
    labFormat = labelFormat(prefix = "$")
  )

# CONVERT THE AMES DATA TO SIMPLE FEATURE FORMAT
ames_sf <- st_as_sf(ames_data, coords = c("longitude", "latitude"), crs = "WGS84")

# CREATE POLYGONS FOR EACH NEIGHBORHOOD BY AGGREGATING POINTS INTO A CONVEX HULL
ames_sf <- ames_sf %>%
  group_by(neighborhood) %>%
  summarize() %>%
  st_convex_hull()

col_vals <- brewer.pal(8, "Dark2")

ames_cols <- c(
  North_Ames = col_vals[3],
  College_Creek = col_vals[4],
  Old_Town = col_vals[8],
  Edwards = col_vals[5],
  Somerset = col_vals[8],
  Northridge_Heights = col_vals[4],
  Gilbert = col_vals[2],
  Sawyer = col_vals[7],
  Northwest_Ames = col_vals[6],
  Sawyer_West = col_vals[8],
  Mitchell = col_vals[3],
  Brookside = col_vals[1],
  Crawford = col_vals[4],
  Iowa_DOT_and_Rail_Road = col_vals[2],
  Timberland = col_vals[2],
  Northridge = col_vals[1],
  Stone_Brook = col_vals[5],
  South_and_West_of_Iowa_State_University = col_vals[3],
  Clear_Creek = col_vals[1],
  Meadow_Village = col_vals[1],
  Briardale = col_vals[7],
  Bloomington_Heights = col_vals[1],
  Veenker = col_vals[2],
  Northpark_Villa = col_vals[2],
  Blueste = col_vals[5],
  Greens = col_vals[6],
  Green_Hills = col_vals[8],
  Landmark = col_vals[3],
  Hayden_Lake = "red"
)

# PLOT POLYGONS
ggplot(ames_sf) +
  ggspatial::annotation_map_tile(type = "cartolight", zoomin = 0) +
  geom_sf(aes(fill = neighborhood), alpha = .2, show.legend = FALSE) +
  scale_fill_manual(values = ames_cols) +
  labs(
    title = "Neighborhoods in Ames, Iowa",
    x = "Longitude",
    y = "Latitude"
  )

# ARE BIGGER HOMES MORE EXPENSIVE
ggplot(data = ames_data %>% mutate(total_sf = first_flr_sf + second_flr_sf), mapping = aes(x = total_sf, y = sale_price)) +
  geom_point(alpha = 0.2) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::dollar) +
  labs(
    title = "Bigger homes are more expensive",
    x = "Square footage",
    y = "Sale price"
  )

# PLOT TOTAL SQUARE FOOT VS SALE PRICE WITH A LINE OF BEST FIT
ggplot(
  data = ames_data %>%
    mutate(total_sf = first_flr_sf + second_flr_sf),
  mapping = aes(x = total_sf, y = sale_price)
) +
  geom_point(alpha = 0.2) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::dollar) +
  labs(
    title = "Are bigger homes more expensive?",
    x = "Square footage",
    y = "Sale price"
  ) +
  stat_poly_line() +
  stat_poly_eq()

# MAHALANOBIS DISTANCE TO REMOVE OUTLIERS
ames_with_tsf <- ames_data %>%
  mutate(total_sf = first_flr_sf + second_flr_sf)

# SELECT ALL NUMERIC COLUMNS
numeric_columns <- ames_with_tsf %>%
  select(where(is.numeric))

# PERFORM PCA TO REDUCE THE DIMENSIONALITY OF THE DATA
pca_result <- prcomp(numeric_columns, scale = TRUE)

# COMPUTE THE NUMBER OF COMPONENTS NEEDED TO EXPLAIN 95% OF THE VARIANCE
var_explained <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
num_components <- which(var_explained > 0.95)[1]
pca_data <- as.data.frame(pca_result$x[, 1:num_components])

# COMPUTE THE MEANS AND COVARIANCE MATRIX FOR THE PCA DATA
means <- colMeans(pca_data)
cov_matrix <- cov(pca_data)

# DEGREES OF FREEDOM FOR THE CHI-SQUARED TEST
dof <- num_components

# SPECIFY THE CRITICAL VALUE FOR THE CHI-SQUARED TEST
alpha <- 0.05

# CALCULATE THE CRITICAL VALUE FOR THE CHI-SQUARED TEST
critical_value <- qchisq(1 - alpha, dof)

# CALCULATE THE MAHALANOBIS DISTANCE FOR EACH OBSERVATION
ames_outlier_ids <- ames_with_tsf %>%
  mutate(
    row_id = row_number(),
    mahalanobis_distance = mahalanobis(
      x = pca_data,
      center = means,
      cov = cov_matrix
    )
  ) %>%
  filter(mahalanobis_distance > critical_value) %>%
  pull(row_id)

# REMOVE OUTLIERS FROM THE ORIGINAL DATA
ames_no_outliers <- ames_data %>%
  filter(!row_number() %in% ames_outlier_ids)

# PLOT THE SALES PRICE AGAINST THE SQUARE FOOTAGE WITH THE OUTLIERS REMOVED
ggplot(data = ames_no_outliers %>% mutate(total_sf = first_flr_sf + second_flr_sf), mapping = aes(x = total_sf, y = sale_price)) +
  geom_point(alpha = 0.2) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::dollar) +
  labs(
    title = "Are bigger homes more expensive? (Outliers removed)",
    x = "Square footage",
    y = "Sale price"
  ) +
  stat_poly_line() +
  stat_poly_eq()

# WHICH NEIGHBOURHOODS HAVE THE HIGHEST PROPORTION OF OVERALL SALES
ames_data %>%
  count(neighborhood) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(pct, reorder(neighborhood, pct))) +
  geom_point() +
  theme_minimal() +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = "Proportion of home sales by Neighbourhood",
    x = "Percent of overall sales",
    y = ""
  )

# WHICH MONTHS SEE THE MOST SALES
ggplot(ames_data, aes(factor(mo_sold))) +
  geom_bar() +
  labs(
    title = "In which month are the most houses sold?",
    x = "Month"
  ) +
  theme_minimal()

# IS THERE A DIFFERENCE IN SALE PRICE BY YEAR SOLD I.E. ARE THE MEDIAN SALE PRICES DIFFERENT
ames_data %>%
  select(neighborhood, year_sold, sale_price) %>%
  group_by(year_sold) %>%
  mutate(
    avg_sale_price = mean(sale_price),
    year_sold = as.factor(year_sold)
  ) %>%
  arrange(year_sold) %>%
  ggplot(aes(x = year_sold, y = sale_price)) +
  geom_boxplot() +
  theme(legend.position = "none") +
  scale_y_continuous(labels = scales::dollar_format()) +
  labs(
    title = "Is there a difference in sale price by year sold?",
    x = "Year Sold",
    y = "Sale Price"
  )

# WHATS THE DISTRIBUTION OF SALE PRICE BY OVERALL QUALITY
ggplot(ames_data, aes(x = sale_price, y = overall_qual, fill = as.factor(overall_qual))) +
  ggridges::geom_density_ridges() +
  scale_x_continuous(labels = dollar) + # Use dollar format for the sale_price axis
  labs(
    title = "Is there a difference in sale price by overall quality?",
    x = "Sale Price",
    y = "Overall Quality"
  ) +
  ggridges::theme_ridges() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5),
    axis.title.x = element_text(hjust = 0.5),
    axis.title.y = element_text(hjust = 0.5)
  )

# WHATS THE RELATIONSHIP BETWEEN SALE PRICE AND THE LAT AND LONG COORDINATES
ames_data %>%
  dplyr::select(sale_price, longitude, latitude) %>%
  tidyr::pivot_longer(
    cols = c(longitude, latitude),
    names_to = "predictor", values_to = "value"
  ) %>%
  ggplot(aes(x = value, sale_price)) +
  geom_point(alpha = .2) +
  geom_smooth(se = FALSE) +
  facet_wrap(~predictor, scales = "free_x") +
  scale_y_continuous(labels = scales::dollar_format()) +
  labs(
    title = "Sales price as a function of location",
    x = "Location Value",
    y = "Sales Price"
  )

# DEFINE PREPROCESSING RECIPES --------------------------------------------

# split into training and testing datasets. Stratify by Sale price
ames_split <- rsample::initial_split(
  ames_no_outliers,
  prop = 0.8,
  strata = sale_price
)

# CREATE TRAINING AND TESTING OBJECTS FROM THE SPLIT OBJECT
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

# CREATE RESAMPLES TO CHOOSE AND COMPARE MODELS
set.seed(234)
ames_folds <- vfold_cv(ames_train, strata = sale_price, v = 10)

# DEFINE PREPROCESSING RECIPES --------------------------------------------

# rand_forest(mode = "regression") %>%
#   set_engine(
#     "ranger",
#     importance = "impurity"
#   ) %>%
#   fit(sale_price ~ .,
#     data = ames_train
#   ) %>%
#   vip::vip()

# BASE RECIPE
base_rec <-
  recipe(sale_price ~ ., data = ames_train) %>%
  step_log(sale_price, base = 10) %>%
  step_YeoJohnson(lot_area, gr_liv_area) %>%
  step_other(neighborhood, threshold = .1) %>%
  step_dummy(all_nominal()) %>%
  step_zv(all_predictors()) %>%
  step_ns(latitude, longitude, deg_free = tune())

# THERE IS 277 COLUMNS IN THE BASE RECIPE
# base_rec %>%
#   prep() %>%
#   juice() %>%
#   ncol()

# RECIPES WITH INTERACTIONS
bt_rec <-
  recipe(sale_price ~ overall_qual + gr_liv_area + bsmt_qual + garage_cars + garage_area + year_built + total_bsmt_sf + exter_qual + first_flr_sf + kitchen_qual, data = ames_train) %>%
  step_log(sale_price, base = 10) %>%
  step_log(gr_liv_area, base = 10) %>%
  #step_other(neighborhood, threshold = 0.05) %>%
  step_dummy(all_nominal_predictors())
  #step_interact(~ gr_liv_area:starts_with("bldg_type_")) %>%
  #step_ns(latitude, longitude, deg_free = tune())

# THERE IS 21 COLUMNS IN THE BOOSTED TREE RECIPE
# bt_rec %>%
#   prep() %>%
#   juice() %>%
#   ncol()

# BUILD MODELS -----------------------------------------------------------

# DEFINE A BAGGED RANDOM FOREST MODEL
bagged_spec <- bag_tree(
  tree_depth = tune(),
  min_n = tune(),
  cost_complexity = tune()
) %>%
  set_mode("regression") %>%
  set_engine("rpart", times = 25L)

# DEFINE A RANGER RANDOM FOREST MODEL
rf_spec <-
  rand_forest(
    mtry = tune(),
    min_n = tune(),
    trees = 1000
  ) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# DEFINE AN XGBOOST MODEL
xgb_spec <- boost_tree(
  trees = 500,
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost", importance = TRUE) %>%
  set_mode("regression")

# DEFINE A BOOSTED TREE ENSEMBLE MODEL
bt_spec <-
  boost_tree(
    learn_rate = tune(),
    stop_iter = tune(),
    trees = 1000
  ) %>%
  set_engine("lightgbm", num_leaves = tune()) %>%
  set_mode("regression")

# DEFINE A WORKFLOW SET ---------------------------------------------------

wflw_set <-
  workflow_set(
    preproc = list(base = base_rec, bt = bt_rec),
    models = list(xgb = xgb_spec, bagged = bagged_spec, rf = rf_spec, bt = bt_spec)
  )

# UPDATE MTRY PARAMETER FOR THE BASE XGBOOST
base_xgb_param <- wflw_set %>%
  extract_workflow(
    id = "base_xgb"
  ) %>%
  hardhat::extract_parameter_set_dials() %>%
  update(mtry = mtry(c(1, 277)))

# UPDATE MTRY PARAMETER FOR THE BASE RF MODEL
base_rf_param <- wflw_set %>%
  extract_workflow(
    id = "base_rf"
  ) %>%
  hardhat::extract_parameter_set_dials() %>%
  update(mtry = mtry(c(1, 277)))

# UPDATE MTRY PARAMETER FOR THE BASE BAGGED MODEL
bt_xgb_param <- wflw_set %>%
  extract_workflow(
    id = "bt_xgb"
  ) %>%
  hardhat::extract_parameter_set_dials() %>%
  update(mtry = mtry(c(1, 21)))

# UPDATE MTRY PARAMETER FOR THE BASE BAGGED MODEL
bt_rf_param <- wflw_set %>%
  extract_workflow(
    id = "bt_rf"
  ) %>%
  hardhat::extract_parameter_set_dials() %>%
  update(mtry = mtry(c(1, 21)))

# UPDATE THE WORKFLOW SET WITH THE NEW PARAMETERS
wf_set_tune_list_finalize <- wflw_set %>%
  option_add(param_info = base_xgb_param, id = "base_xgb") %>%
  option_add(param_info = base_rf_param, id = "base_rf") %>%
  option_add(param_info = bt_xgb_param, id = "bt_xgb") %>%
  option_add(param_info = bt_rf_param, id = "bt_rf")

# SPECIFY THE TUNE GRID
race_ctrl <-
  control_race(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
  )

# REGISTER THE PARALLEL BACKEND
doParallel::registerDoParallel(cores = 4)

# ADD NEW PARAMETERS TO THE WORKFLOW SET
tictoc::tic()
race_results <- wf_set_tune_list_finalize %>%
  workflow_map(
    "tune_race_anova",
    seed = 123,
    resamples = ames_folds,
    grid = 25,
    control = race_ctrl,
    verbose = TRUE
  )
tictoc::toc()

# STOP THE PARALLEL BACKEND
doParallel::stopImplicitCluster()

race_results

# EVALUATE THE MODELS 
autoplot(race_results,
         rank_metric = "rmse",
         metric = "rmse",
         select_best = TRUE) +
  geom_text(aes(y = (mean - std_err) - std_err, label = wflow_id), angle = 90, hjust = 1) +
  #lims(y = c(min(race_results$mean, max(race_results$mean)))) +
  theme(legend.position = "none")

best_results <- 
  race_results %>% 
  extract_workflow_set_result("base_xgb") %>% 
  select_best(metric = "rmse")

best_results

test_results <- 
  race_results %>% 
  extract_workflow("base_xgb") %>% 
  finalize_workflow(best_results) %>% 
  last_fit(split = ames_split)

collect_metrics(test_results)

# VISUALIZE THE TEST SET RESULTS
test_results %>% 
  collect_predictions() %>% 
  ggplot(aes(x = sale_price, y = .pred)) + 
  geom_abline(color = "gray50", lty = 2) + 
  geom_point(alpha = 0.5) + 
  coord_obs_pred() + 
  labs(x = "observed (log10)", y = "predicted (log10)", title = "Observed Vs. Predicted Sale Price")

# PLOT THE DISTRIBUTION OF PREDICTION ERRORS
test_results %>%
  collect_predictions() %>%
  mutate(error = sale_price - .pred) %>% # Calculate the errors
  ggplot(aes(x = error)) + # Plot the errors
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "blue", alpha = 0.5) + # Histogram with density scaling
  geom_density(color = "red", linewidth = 1) + # Density plot overlay
  labs(x = "Error (sale_price - prediction)", 
       y = "Density", 
       title = "Distribution of Prediction Errors with Density Overlay") +
  theme_minimal()

test_results %>%
  collect_predictions() %>%
  arrange(.pred) %>%
  mutate(residual_pct = (sale_price - .pred) / .pred) %>%
  select(.pred, residual_pct) %>% 
  ggplot(., aes(x = .pred, y = residual_pct)) +
  geom_point() +
  labs(x = "Predicted Sale Price", y = "Residual (%)", title = "Residuals for Test Data") +
  scale_x_continuous(labels = scales::dollar_format()) +
  scale_y_continuous(labels = scales::percent)
