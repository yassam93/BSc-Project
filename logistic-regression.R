# install libraries
# install.packages("dotwhisker")

req <- substitute(require(x, character.only = TRUE))
libs<-c("sjPlot", "ggplot2", "jtools", "car", "blorr", "DescTools", "MASS",
        "dotwhisker", "interactions")
sapply(libs, function(x) eval(req) || {install.packages(x); eval(req)})


# load libraries
library(dplyr)
library(FSA)
library(PerformanceAnalytics)
library(caTools)
library(ROCR)
library(psych)
library(readr)
library(glmnet)
library(dotwhisker)
library(interactions)
library(coefplot)
library(olsrr)
library(stargazer)
library(broom.mixed)
library(visreg)
library(aod)
library(caret)
library(lmtest)
# library(tidymodels)
library(foreign)
library(nnet)
library(reshape2)
# library(ggplot2)
library(modelsummary)

# load dataset
Data = read.table("datasets/preprocessed_dataset.csv", header=TRUE, 
                  stringsAsFactors=TRUE, sep = "\t")
# load dataset head
head(Data)


# get columns
columns = names(Data)

# convert dataset to numeric
Data.num =
  select(Data,
         Age,
         Sleep_Duration,
         Quality_of_Sleep,
         Physical_Activity_Level,
         Stress_Level,
         BMI_Category,
         Blood_Pressure,
         Heart_Rate,
         Daily_Steps,
         Female,
         Male,
         Accountant,
         Doctor,
         Engineer,
         Lawyer,
         Manager,
         Nurse,
         Sales_Representative,
         Salesperson,
         Scientist,
         Software_Engineer,
         Teacher,
         Sleep_Disorder)

summary(Data.num)



# convert DATA.num columns to numeric
Data.num$Age                      = as.numeric(Data.num$Age)
Data.num$Sleep_Duration           = as.numeric(Data.num$Sleep_Duration)
Data.num$Quality_of_Sleep         = as.numeric(Data.num$Quality_of_Sleep)
Data.num$Physical_Activity_Level  = as.numeric(Data.num$Physical_Activity_Level)
Data.num$Stress_Level             = as.numeric(Data.num$Stress_Level)
Data.num$BMI_Category             = as.numeric(Data.num$BMI_Category)
Data.num$Blood_Pressure           = as.numeric(Data.num$Blood_Pressure)
Data.num$Heart_Rate               = as.numeric(Data.num$Heart_Rate)
Data.num$Daily_Steps              = as.numeric(Data.num$Daily_Steps)
Data.num$Female                   = as.numeric(Data.num$Female)
Data.num$Male                     = as.numeric(Data.num$Male)
Data.num$Accountant               = as.numeric(Data.num$Accountant)
Data.num$Doctor                   = as.numeric(Data.num$Doctor)
Data.num$Engineer                 = as.numeric(Data.num$Engineer)
Data.num$Lawyer                   = as.numeric(Data.num$Lawyer)
Data.num$Manager                  = as.numeric(Data.num$Manager)
Data.num$Nurse                    = as.numeric(Data.num$Nurse)
Data.num$Sales_Representative     = as.numeric(Data.num$Sales_Representative)
Data.num$Salesperson              = as.numeric(Data.num$Salesperson)
Data.num$Scientist                = as.numeric(Data.num$Scientist)
Data.num$Software_Engineer        = as.numeric(Data.num$Software_Engineer)
Data.num$Teacher                  = as.numeric(Data.num$Teacher)
Data.num$Sleep_Disorder           = as.numeric(Data.num$Sleep_Disorder)

#Data.num$Sleep_Disorder <- factor(Data.num$Sleep_Disorder)
#class(Data.num$Sleep_Disorder)
#train_dataset$Sleep_Disorder <- factor(train_dataset$Sleep_Disorder)
#class(train_dataset$Sleep_Disorder)

# view dataset
View(Data)
View(Data.num)
headTail(Data.num)


# check correlations among variables
chart.Correlation(Data.num,
                  method="spearman",
                  histogram=TRUE,
                  pch=24)


# correlation_matrix = corr.test(Data.num, use = "pairwise", method="spearman", adjust="none", alpha=.05)
corr.test(Data.num, use = "pairwise", method="spearman", adjust="none", alpha=.05)



Data$Sleep_Disorder[Data$Sleep_Disorder==0] <- "None"
Data$Sleep_Disorder[Data$Sleep_Disorder==1] <- "Sleep_Apnea"
Data$Sleep_Disorder[Data$Sleep_Disorder==2] <- "Insomnia"
class(Data$Sleep_Disorder)
class(Data.num$Sleep_Disorder)

#Factorising sleep disorder due to our problem being Classification
Data.num$Sleep_Disorder <- factor(Data$Sleep_Disorder)
class(Data.num$Sleep_Disorder)
#train_dataset$Sleep_Disorder <- factor(train_dataset$Sleep_Disorder)
#class(train_dataset$Sleep_Disorder)

# We take one point in sleep disorder as reference as 'None'
# And compare Insomnia and Sleep Apnea according to this reference
Data.num$Base_Sleep_Disorder <- relevel(factor(Data.num$Sleep_Disorder), ref = "None")

# Spit data into train and test:
# https://www.geeksforgeeks.org/split-the-dataset-into-the-training-test-set-in-r/
# https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function
# https://stackoverflow.com/questions/66654264/data-splitting-into-training-and-testing-data


filter <- sample(c(TRUE,FALSE), nrow(Data.num), replace=TRUE, prob=c(0.8,0.2))
train_dataset  <- Data.num[filter, ]
test_dataset  <- Data.num[!filter, ]  


# convert Sleep_Disorder to factor
#class(train_dataset$Sleep_Disorder)
#train_dataset$Sleep_Disorder <- factor(train_dataset$Sleep_Disorder)
#class(train_dataset$Sleep_Disorder)

# Base sleep disorder which None was referenced in it, 
# And other sleep disorder values will be compared base on this
class(train_dataset$Base_Sleep_Disorder )

# creating our logistic regression model, 
# which is gonna estimate Base_Sleep_Disorder here, from these given features
# to check which ones are more relevant
multinomial_logistic_regression <- multinom(Base_Sleep_Disorder ~ Age + Sleep_Duration + Quality_of_Sleep + 
                                              Physical_Activity_Level + Stress_Level + BMI_Category + 
                                              Heart_Rate + Daily_Steps + Female + Male + Accountant + Doctor +
                                              Engineer + Lawyer + Nurse + Sales_Representative +
                                              Salesperson + Scientist + Software_Engineer + Teacher + 
                                              Blood_Pressure, data = train_dataset)


summary(multinomial_logistic_regression)

z <- summary(multinomial_logistic_regression)$coefficients/summary(multinomial_logistic_regression)$standard.errors

# If p value is small it is desirable for us
p <- (1 - pnorm(abs(z), 0, 1)) * 2

# exp(cbind(OR = coef(multinomial_logistic_regression), confint(multinomial_logistic_regression))) 
# results in ORs and their CIs


exp(coef(multinomial_logistic_regression))
head(pp <- fitted(multinomial_logistic_regression))
confint(multinomial_logistic_regression)
# examining accuracy model on test data
predict(multinomial_logistic_regression, newdata = test_dataset, "probs")
predict(multinomial_logistic_regression, newdata = test_dataset, interval="prediction")
prediction <- predict(multinomial_logistic_regression, newdata = test_dataset, "probs")
prediction <- cbind(1:nrow(prediction), max.col(prediction, 'first'))
prediction <- prediction[, 2] - 1
prediction[prediction ==0] <- "None"
prediction[prediction ==2] <- "Sleep_Apnea"
prediction[prediction ==1] <- "Insomnia"
test_dataset$predict <- prediction

# confusion matrix to realise the accuracy on each class
t1 <- test_dataset[c("Base_Sleep_Disorder","predict")]
table(t1)
t1$Base_Sleep_Disorder <- as.character(t1$Base_Sleep_Disorder)
t1$predict <- as.character(t1$predict)



conf_mat_tab <- table(lapply(t1, factor))
confusionMatrix(conf_mat_tab)



# starting from simpler model null to the model with more features full, to find out which features to keep

multinomial_logistic_regression_null = multinom(Base_Sleep_Disorder ~ 1,
                                                data=train_dataset)

multinomial_logistic_regression_full <- multinom(Base_Sleep_Disorder ~ Age + Sleep_Duration + Quality_of_Sleep + 
                                                   Physical_Activity_Level + Stress_Level + BMI_Category + 
                                                   Heart_Rate + Daily_Steps + Female + Male + Accountant + Doctor +
                                                   Engineer + Lawyer + Nurse + Sales_Representative +
                                                   Salesperson + Scientist + Software_Engineer + Teacher + 
                                                   Blood_Pressure, data = train_dataset)


# Applying each feature step by step, from null getting to full, 
# to decide which features to keep so that we can choose our model then
step(multinomial_logistic_regression_null,
     scope = list(upper=multinomial_logistic_regression_full),
     direction="both",
     test="Chisq",
     data=train_dataset)



# Final model picked for logistic regression based on output of previous step
multinomial_logistic_regression_final <- multinom(formula = Base_Sleep_Disorder ~ Nurse + Heart_Rate + 
                                                    Sleep_Duration + Doctor + Daily_Steps + Scientist + Teacher + 
                                                    Age + Sales_Representative + Salesperson, data = train_dataset)


#Summary of the model
summary(multinomial_logistic_regression_final)

#Statistical test to estimate how a quantitative dependent variable 
#changes based on levels of one or more categorical independent variables.
#Available at: https://www.scribbr.com/statistics/anova-in-r/

Anova(multinomial_logistic_regression_final, type="II", test="Wald")

# We wanna examine the model with the output recommended features
Data.final =
  select(train_dataset,
         Sales_Representative,
         Daily_Steps,
         Teacher,
         Doctor,
         Scientist,
         Nurse,
         Heart_Rate,
         Age,
         Sleep_Duration,
         Salesperson,
         Base_Sleep_Disorder
  )


# Overall p-value for model, Define null models and compare to final model
multinomial_logistic_regression_null = multinom(Base_Sleep_Disorder ~ 1,
                                                data=Data.final)

# To see what changes we would have, and overall p-value for model
anova(multinomial_logistic_regression_final,
      multinomial_logistic_regression_null,
      test="Chisq")


# Likelihood ratio test
lrtest(multinomial_logistic_regression_final)

summary(multinomial_logistic_regression_final)

# Test model again based on selected features
predict(multinomial_logistic_regression_final, newdata = test_dataset, "probs")
predict(multinomial_logistic_regression_final, newdata = test_dataset, interval="prediction")
prediction <- predict(multinomial_logistic_regression_final, newdata = test_dataset, "probs")
prediction <- cbind(1:nrow(prediction), max.col(prediction, 'first'))
prediction <- prediction[, 2] - 1
prediction[prediction ==0] <- "None"
prediction[prediction ==2] <- "Sleep_Apnea"
prediction[prediction ==1] <- "Insomnia"
test_dataset$predict <- prediction

#Confusion matrix
t2 <- test_dataset[c("Base_Sleep_Disorder","predict")]
table(t2)
t2$Base_Sleep_Disorder <- as.character(t2$Base_Sleep_Disorder)
t2$predict <- as.character(t2$predict)



conf_mat_tab_2 <- table(lapply(t2, factor))
confusionMatrix(conf_mat_tab_2)

# After simplifying the model with choosing less features, reesults of accuracy 
# was not so different with conf_mat_tab, therefore keeping this model

# Comparing based on tests which each one increments with one feature, 
# Based on how the p-value differs, and how the performance of model could improve. 
multinomial_logistic_regression_1 <- multinom(Base_Sleep_Disorder ~ 1, data = train_dataset)
multinomial_logistic_regression_2 <- multinom(Base_Sleep_Disorder ~ Sales_Representative, data = train_dataset)
multinomial_logistic_regression_3 <- multinom(Base_Sleep_Disorder ~ Sales_Representative + Daily_Steps, data = train_dataset)
multinomial_logistic_regression_4 <- multinom(Base_Sleep_Disorder ~ Sales_Representative + Daily_Steps +
                                                Doctor, data = train_dataset)
multinomial_logistic_regression_5 <- multinom(Base_Sleep_Disorder ~ Sales_Representative + Daily_Steps +
                                                Doctor + Teacher, data = train_dataset)
multinomial_logistic_regression_6 <- multinom(Base_Sleep_Disorder ~ Sales_Representative + Daily_Steps +
                                                Doctor + Teacher + Scientist, data = train_dataset)
multinomial_logistic_regression_7 <- multinom(Base_Sleep_Disorder ~ Sales_Representative + Daily_Steps +
                                                Doctor + Teacher + Scientist + Nurse, data = train_dataset)
multinomial_logistic_regression_8 <- multinom(Base_Sleep_Disorder ~ Sales_Representative + Daily_Steps +
                                                Doctor + Teacher + Scientist + Nurse + Heart_Rate, data = train_dataset)
multinomial_logistic_regression_9 <- multinom(Base_Sleep_Disorder ~ Sales_Representative + Daily_Steps +
                                                Doctor + Teacher + Scientist + Nurse + Heart_Rate +
                                                Age, data = train_dataset)
multinomial_logistic_regression_10 <- multinom(Base_Sleep_Disorder ~ Sales_Representative + Daily_Steps +
                                                 Doctor + Teacher + Scientist + Nurse + Heart_Rate +
                                                 Age + Sleep_Duration, data = train_dataset)
multinomial_logistic_regression_11 <- multinom(Base_Sleep_Disorder ~ Sales_Representative + Daily_Steps +
                                                 Doctor + Teacher + Scientist + Nurse + Heart_Rate +
                                                 Age + Sleep_Duration + Salesperson, data = train_dataset)




anova(multinomial_logistic_regression_1, multinomial_logistic_regression_2,
      multinomial_logistic_regression_3, multinomial_logistic_regression_4,
      multinomial_logistic_regression_5, multinomial_logistic_regression_6,
      multinomial_logistic_regression_7, multinomial_logistic_regression_8,
      multinomial_logistic_regression_9, multinomial_logistic_regression_10,
      multinomial_logistic_regression_11, test="Chisq")
