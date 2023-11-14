library(tidymodels)
library(vroom)
library(caret)
library(dplyr)
library(randomForest)
library(gbm)

ggg_train <- vroom("/Users/cicizeng/Desktop/STA348/Ghosts, Ghouls and Goblings/train.csv") 
ggg_test <- vroom("/Users/cicizeng/Desktop/STA348/Ghosts, Ghouls and Goblings/test.csv") 

# Feature Engineering: Create interaction terms
ggg_train <- ggg_train %>%
  mutate(hair_soul = hair_length * has_soul,
         bone_flesh = bone_length * rotting_flesh,
         bone_hair = bone_length * hair_length,
         bone_soul = bone_length * has_soul,
         flesh_hair = rotting_flesh * hair_length,
         flesh_soul = rotting_flesh * has_soul)

ggg_test <- ggg_test %>%
  mutate(hair_soul = hair_length * has_soul,
         bone_flesh = bone_length * rotting_flesh,
         bone_hair = bone_length * hair_length,
         bone_soul = bone_length * has_soul,
         flesh_hair = rotting_flesh * hair_length,
         flesh_soul = rotting_flesh * has_soul)

# Apply PCA
pca_model <- preProcess(ggg_train[, -which(names(ggg_train) %in% c("id", "type"))], method = "pca", pcaComp = 5)
train_pca <- predict(pca_model, ggg_train)
test_pca <- predict(pca_model, ggg_test)

# Combine PCA components with the original data
train_pca$type <- ggg_train$type
test_pca$id <- ggg_test$id

# Train Control for cross-validation
myControl <- trainControl(method = "cv", number = 10)

# Hyperparameter Tuning for Random Forest
set.seed(10)
rf_model <- train(
  type ~ ., 
  data = train_pca, 
  method = "rf", 
  trControl = myControl,
  tuneLength = 5  # Increase tune length for more options
)

# Updated GBM Model with expanded tuneGrid
set.seed(10)
gbm_model <- train(
  type ~ ., 
  data = train_pca, 
  method = "gbm", 
  trControl = myControl,
  verbose = FALSE,
  tuneGrid = expand.grid(
    interaction.depth = 1:5,             # Increased depth
    n.trees = seq(50, 200, by = 50),     # More trees
    shrinkage = c(0.05, 0.1, 0.15),      # Different shrinkage rates
    n.minobsinnode = c(5, 10, 20)        # Different values for min. observations
  )
)

# Model comparison
models <- list(rf = rf_model, gbm = gbm_model)
resampled <- resamples(models)
summary(resampled)

# Choose the best model and predict on test set
# [Select the best model based on the summary]
# For example, assuming gbm_model performs better
predicted_class <- predict(gbm_model, test_pca)

# Prepare the submission file
my_solution <- data.frame(id = test_pca$id, type = predicted_class)

# Write the submission data frame to a CSV file
vroom_write(x = my_solution, file = "./ggg.csv", delim = ",")

