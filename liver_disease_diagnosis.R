library(tidyverse)
library(data.table)
library(caret)

#  Downloading the dataset zip file
dl <- tempfile()
download.file("https://github.com/telyon/liver_disease_diagnosis/raw/main/Liver%20patients%20data.zip", dl)
#  Extract to current working directory
extract <- unzip(dl)
#  load the data to a variable
data <- read.csv(extract)

#  wrangling

colnames(data) <- c("age", "gender", "tot_bil", "direct_bil", "alkal_phos", 
                    "alam_amino", "aspa_amino", "tot_proteins", "albu", 
                    "albu_glo_ratio", "diagnosis")
data <- data %>% mutate(gender = as.factor(gender),
                        diagnosis = as.factor(ifelse(str_detect(diagnosis, "1"), "pos", "neg")))

# any missing data?
any(is.na(data))
missing <- which(is.na(data))
# variables (columns) with missing data
ceiling(missing/nrow(data))
##INF::: 4 values missing from column 10 (albu_glo_ratio) only
# rows with missing value
missing <- which(is.na(data$albu_glo_ratio))
# remove the 4 rows with missing albu_glo_ratio value
data_clean <- data[-(which(is.na(data$albu_glo_ratio))),]
# clean up
rm(data, dl, extract, missing)

###  exploratory analysis
str(data_clean)

#  age range
range(data_clean$age)

#  age distribution
hist(data_clean$age)

#  check proportion of sample that has liver disease
mean(data_clean$diagnosis == "pos")
##INF::: There seems to be a high prevalence of liver disease in the dataset: about 71%

#  does age affect likelihood of liver disease?
plot(data_clean$age, data_clean$diagnosis)

#  are there equal number of both gender?
table(data_clean$gender)

#  does gender affect likelihood of liver disease?
ggplot(data_clean, aes(diagnosis, fill = gender))+
  geom_bar(position = "dodge")

#  proportion of each gender that has liver diseases
count(data_clean, c("gender", "diagnosis")) %>% 
  spread(diagnosis, freq) %>% 
  mutate(disease_prop = pos/(pos+neg))

#  investigating variations in liver content with age, and gender, and if they
#  affect likelihood of liver disease
data_clean %>% gather("component", "value", tot_bil, direct_bil, alkal_phos,
                alam_amino, aspa_amino, tot_proteins, albu, albu_glo_ratio) %>%
  ggplot(aes(age, value, col = diagnosis)) +
  geom_point() +
  facet_wrap(~component)

data_clean %>% gather("component", "value", tot_bil, direct_bil, alkal_phos,
                      alam_amino, aspa_amino, tot_proteins, albu, albu_glo_ratio) %>%
  ggplot(aes(gender, value, col = diagnosis)) +
  geom_point(position = "jitter") +
  facet_wrap(~component)


# investigate a possible relationship between aspa_amino level and liver disease
data_clean %>% ggplot(aes(aspa_amino, diagnosis)) +
  geom_point(position = "jitter") +
  scale_x_log10()
##INF::: there is no clear distinction between the levels of
# aspa_amino in patients with positive and negative diagnosis.

# investigate a possible relationship between alam_amino level and liver disease
data_clean %>% ggplot(aes(alam_amino, diagnosis)) +
  geom_point(position = "jitter") +
  scale_x_log10()
##INF::: there is no clear distinction between the levels of
# alam_amino in patients with positive and negative diagnosis.

# investigate a possible relationship between alkal_phos level and liver disease
data_clean %>% ggplot(aes(alkal_phos, diagnosis)) +
  geom_point(position = "jitter") +
  scale_x_log10()
##INF::: there is no clear distinction between the levels of
# alkal_phos in patients with positive and negative diagnosis.

### Building prediction model
 

#  partitioning the data into training and validation sets
set.seed(35, sample.kind = "Rounding")
sampler <- createDataPartition(data_clean$diagnosis, times=1, p=0.1, list = F)
validation <- data_clean[sampler, ]
training_set <- data_clean[-sampler, ]

#  random prediction of diagnosis
guess <- sample(c("neg", "pos"), nrow(training_set), replace = TRUE)
guess_accuracy <- mean(guess == training_set$diagnosis)
#  store accuracy in a table
results <- data.frame("method" = "guessing", "accuracy" = guess_accuracy)

# taking into account the higher prevalence of liver disease, we could increase
# the probability of a positive diagnosis in the guess: using 0.7, the proportion of the sample with liver disease
mod_guess <- sample(c("neg", "pos"), nrow(training_set), replace = TRUE, prob = c(0.3, 0.7))
mod_guess_accuracy <- mean(mod_guess == training_set$diagnosis)
results <- bind_rows(results, data.frame("method" = "guessing with prevalence", "accuracy" = mod_guess_accuracy))
results
##INF::: Incorporating prevalnce into the guess produced only marginal improvement in the accuracy of the guess

#  partitioning the training data into and train and test sets
sampler2 <- createDataPartition(training_set$diagnosis, times = 1, p = 0.2, list = F)
test_set <- training_set[sampler2, ]
train_set <- training_set[-sampler2, ]

#  using K-nearest neighbours (KNN) algorithm
y_hat_knn <- predict(train(diagnosis ~ ., data = train_set, method = "knn"), test_set)
knn_accuracy <- mean(y_hat_knn == test_set$diagnosis)
results <- bind_rows(results, data.frame("method" = "knn", "accuracy" = knn_accuracy))
results
##INF::: Accuracy seems to be much better with the KNN algorithm than the guess

#  using logistic regression (GLM) algorithm
y_hat_glm <- predict(train(diagnosis ~ ., data = train_set, method = "glm"), test_set)
glm_accuracy <- mean(y_hat_glm == test_set$diagnosis)
results <- bind_rows(results, data.frame("method" = "glm", "accuracy" = glm_accuracy))
results

#  using linear discriminant analysis (LDA) algorithm
y_hat_lda <- predict(train(diagnosis ~ ., data = train_set, method = "lda"), test_set)
lda_accuracy <- mean(y_hat_lda == test_set$diagnosis)
results <- bind_rows(results, data.frame("method" = "lda", "accuracy" = lda_accuracy))
results

#  using quadratic discriminant analysis (QDA) algorithm
y_hat_qda <- predict(fit_qda <- train(diagnosis ~ ., data = train_set, method = "qda"), test_set)
qda_accuracy <- mean(y_hat_qda == test_set$diagnosis)
results <- bind_rows(results, data.frame("method" = "qda", "accuracy" = qda_accuracy))
results

#  using random forest algorithm
y_hat_rf <- predict(train(diagnosis ~ ., data = train_set, method = "rf", tuneGrid = data.frame(mtry = 2), nodesize = 26), test_set)
rf_accuracy <- mean(y_hat_rf == test_set$diagnosis)
results <- bind_rows(results, data.frame("method" = "random forest", "accuracy" = rf_accuracy))
results

### Creating an Ensemble INF::: from the final results table of algorithm
##accuracy, it is evident that the QDA algorithm preformed even worse than the
##guessing improved with prevalence. So we can ignore the QDA algorithm
##altogether in our ensemble. We can also ignore the KNN algorithm as it does
##not perform so much better than the improved guessing model

#  Assemble the predictions form the different algorithms into a dataframe
preds <- data.frame( "glm" = y_hat_glm, "lda" = y_hat_lda, "rf" = y_hat_rf)
#  get a prediction from the ensemble : predict positive for 2/more positives
preds <- data.frame(ifelse(preds == "pos", 1, 0)) %>% mutate(ens_pred = ifelse((glm+lda+rf) >= 2, "pos", "neg"))
ens_accuracy <- mean(preds$ens_pred == test_set$diagnosis)
results <- bind_rows(results, data.frame("method" = "algorithm ensemble", "accuracy" = ens_accuracy))
results

#  FINAL Evaluation of the algorithm using the whole training-set to train the algorithm and validation as the test set

#  chosen algorithms based on the analysis done
algorithms <- c("glm", "lda", "rf")

#  function to ensemble predictions with chosen algorithms
predicted <- function(method) {
  predict(train(diagnosis ~ ., training_set, method = algorithms), validation)
}

final_preds <- data.frame(sapply(algorithms, predicted))
final_preds <- data.frame(ifelse(final_preds == "pos", 1, 0)) %>% mutate(ens_pred = ifelse((glm+lda+rf) >= 2, "pos", "neg"))
ens_accuracy <- mean(final_preds$ens_pred == validation$diagnosis)
ens_accuracy

# Compare result with using the Logistic regression model with training-set and validation
mean(predict(train(diagnosis ~ ., training_set, method = "glm"), validation) == validation$diagnosis)