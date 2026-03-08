# ==========================================
# CANCER PREDICTION PROJECT
# Decision Tree + Random Forest
# ==========================================


# ==========================================
# 1. LOAD REQUIRED LIBRARIES
# ==========================================

library(rpart)
library(randomForest)
library(rpart.plot)
library(ggplot2)


# ==========================================
# 2. LOAD DATASET
# ==========================================

df <- read.csv("C:/Users/Asus/Downloads/cancer.csv")

# Remove ID column if present
if("id" %in% colnames(df)){
  df$id <- NULL
}

# Convert diagnosis to factor
df$diagnosis <- as.factor(df$diagnosis)


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

set.seed(123)

index <- sample(1:nrow(df), 0.7*nrow(df))

trainData <- df[index, ]
testData  <- df[-index, ]


# ==========================================
# 4. DECISION TREE MODEL
# ==========================================

dt_model <- rpart(diagnosis ~ ., data=trainData, method="class")

dt_pred <- predict(dt_model, testData, type="class")

dt_accuracy <- sum(dt_pred == testData$diagnosis) / length(dt_pred)


# ==========================================
# 5. RANDOM FOREST MODEL
# ==========================================

rf_model <- randomForest(diagnosis ~ ., data=trainData)

rf_pred <- predict(rf_model, testData)

rf_accuracy <- sum(rf_pred == testData$diagnosis) / length(rf_pred)


# ==========================================
# 6. DECISION TREE GRAPH
# ==========================================

rpart.plot(dt_model,
           main="Decision Tree for Cancer Prediction",
           box.palette="RdYlGn",
           shadow.col="gray")


# ==========================================
# 7. FEATURE IMPORTANCE GRAPH
# ==========================================

importance_values <- importance(rf_model)

importance_df <- data.frame(
  Feature = rownames(importance_values),
  Importance = importance_values[,1]
)

importance_df <- importance_df[order(importance_df$Importance,decreasing=TRUE),]

ggplot(importance_df[1:10,],
       aes(x=reorder(Feature,Importance),y=Importance)) +
  geom_bar(stat="identity", fill="steelblue") +
  coord_flip() +
  labs(title="Top 10 Important Features",
       x="Features",
       y="Importance") +
  theme_minimal()


# ==========================================
# 8. MODEL ACCURACY GRAPH
# ==========================================

accuracy_df <- data.frame(
  Model=c("Decision Tree","Random Forest"),
  Accuracy=c(dt_accuracy*100, rf_accuracy*100)
)

ggplot(accuracy_df,
       aes(x=Model,y=Accuracy,fill=Model)) +
  geom_bar(stat="identity",width=0.6) +
  labs(title="Model Accuracy Comparison",
       y="Accuracy (%)") +
  theme_minimal()


# ==========================================
# 9. NEW PATIENT PREDICTION
# ==========================================

new_patient <- data.frame(
  radius_mean = 14.5,
  texture_mean = 20,
  perimeter_mean = 95,
  area_mean = 650,
  smoothness_mean = 0.10,
  compactness_mean = 0.15,
  concavity_mean = 0.12,
  concave.points_mean = 0.08,
  symmetry_mean = 0.20,
  fractal_dimension_mean = 0.06,
  radius_se = 0.5,
  texture_se = 1,
  perimeter_se = 3,
  area_se = 40,
  smoothness_se = 0.005,
  compactness_se = 0.02,
  concavity_se = 0.03,
  concave.points_se = 0.01,
  symmetry_se = 0.02,
  fractal_dimension_se = 0.003,
  radius_worst = 16,
  texture_worst = 25,
  perimeter_worst = 105,
  area_worst = 800,
  smoothness_worst = 0.14,
  compactness_worst = 0.25,
  concavity_worst = 0.30,
  concave.points_worst = 0.12,
  symmetry_worst = 0.30,
  fractal_dimension_worst = 0.08
)

dt_result <- predict(dt_model,new_patient,type="class")
rf_result <- predict(rf_model,new_patient)

rf_prob <- predict(rf_model,new_patient,type="prob")


# ==========================================
# 10. FINAL REPORT
# ==========================================

cat("\n=====================================\n")
cat("        CANCER PREDICTION REPORT\n")
cat("=====================================\n\n")

cat("Decision Tree Accuracy :", round(dt_accuracy*100,2), "%\n")
cat("Random Forest Accuracy :", round(rf_accuracy*100,2), "%\n\n")

cat("Decision Tree Prediction :", dt_result, "\n")
cat("Random Forest Prediction :", rf_result, "\n\n")

cat("Prediction Probability:\n")
print(rf_prob)

cat("\n=====================================\n")
cat("Project Completed Successfully\n")
cat("=====================================\n")