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
# library(tidymodels)

# load dataset
Data = read.table("F:\\Thesis\\preprocessed_dataset.csv", header=TRUE, 
                  stringsAsFactors=TRUE, sep = "\t")
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



# convert columns to numeric
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
corr.test(Data.num, use = "pairwise", method="spearman", adjust="none", 
          alpha=.05)


# Split data into train and test:
# https://www.geeksforgeeks.org/split-the-dataset-into-the-training-test-set-in-r/
# https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function
# https://stackoverflow.com/questions/66654264/data-splitting-into-training-and-testing-data


filter <- sample(c(TRUE,FALSE), nrow(Data.num), replace=TRUE, prob=c(0.8,0.2))
train_dataset  <- Data.num[filter, ]
test_dataset  <- Data.num[!filter, ]  


# convert Sleep_Disorder from numeric to factor
class(train_dataset$Sleep_Disorder)
train_dataset$Sleep_Disorder <- factor(train_dataset$Sleep_Disorder)
class(train_dataset$Sleep_Disorder)