library(tidymodels)
library(vroom)
library(caret)
library(dplyr)
library(xgboost)

train_complete <- vroom("/Users/cicizeng/Desktop/STA348/Ghosts, Ghouls and Goblings/train.csv") 
test_complete <- vroom("/Users/cicizeng/Desktop/STA348/Ghosts, Ghouls and Goblings/test.csv") 
# Feature Engineering: Create interaction terms
train_complete <- train_complete %>%
  mutate(hair_soul = hair_length * has_soul,
         bone_flesh = bone_length * rotting_flesh,
         bone_hair = bone_length * hair_length,
         bone_soul = bone_length * has_soul,
         flesh_hair = rotting_flesh * hair_length,
         flesh_soul = rotting_flesh * has_soul)

test_complete <- test_complete %>%
  mutate(hair_soul = hair_length * has_soul,
         bone_flesh = bone_length * rotting_flesh,
         bone_hair = bone_length * hair_length,
         bone_soul = bone_length * has_soul,
         flesh_hair = rotting_flesh * hair_length,
         flesh_soul = rotting_flesh * has_soul)

# Apply PCA
preProcess_pca <- preProcess(train_complete[, -which(names(train_complete) == "type")], 
                             method = "pca", pcaComp = "variance", thresh = 0.95) # Retain 95% variance
train_pca <- predict(preProcess_pca, train_complete)
test_pca <- predict(preProcess_pca, test_complete)
train_pca$type <- train_complete$type
# Update the target variable to be included in the PCA-transformed data
train_pca$type <- train_complete$type

# Train Control for cross-validation
myControl <- trainControl(method = "cv", number = 10)

# Random Forest Model with PCA-transformed data
set.seed(10)
rf_model <- train(
  type ~ .,  
  data = train_pca, 
  method = "ranger", 
  trControl = myControl,
  importance = 'impurity',
  tuneLength = 3
)

# GLMnet Model with PCA-transformed data
set.seed(10)
glm_model <- train(
  type ~ .,  
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0:1, lambda = seq(0.0001, 1, length = 20)),
  data = train_pca,
  trControl = myControl
)

# Model comparison
models <- list(rf = rf_model, glmnet = glm_model)
resampled <- resamples(models)
summary(resampled)

# Choose the best model and predict on test set
predicted_class <- predict(glm_model, test_pca) # assuming glmnet is better

# Prepare the submission file
my_solution <- data.frame(id = test_complete$id, Type = predicted_class)

# Write the submission data frame to a CSV file
vroom_write(x = my_solution, file = "./ggg.csv", delim = ",")

