# install libraries
# install.packages("dotwhisker")

set.seed(10)

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
library(foreign)
library(nnet)
library(reshape2)
# library(ggplot2)
library(modelsummary)
library(lmtest)
library(caret)
library(modelsummary)
library(car)
library(data.table)
library(kernlab)
library(e1071)  



# load dataset
Data = read.table("datasets/preprocessed_dataset.csv", header=TRUE, 
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
Data.num$Sleep_Disorder           = as.numeric(Data.num$Sleep_Disorder)

#Data.num$Sleep_Disorder <- factor(Data.num$Sleep_Disorder)
#class(Data.num$Sleep_Disorder)
#train_dataset$Sleep_Disorder <- factor(train_dataset$Sleep_Disorder)
#class(train_dataset$Sleep_Disorder)

# view dataset
View(Data)
View(Data.num)
headTail(Data.num)


Data$Sleep_Disorder[Data$Sleep_Disorder==0] <- "None"
Data$Sleep_Disorder[Data$Sleep_Disorder==1] <- "Sleep_Apnea"
Data$Sleep_Disorder[Data$Sleep_Disorder==2] <- "Insomnia"
class(Data$Sleep_Disorder)
class(Data.num$Sleep_Disorder)

Data.num$Sleep_Disorder <- factor(Data$Sleep_Disorder)
class(Data.num$Sleep_Disorder)


Data.num$Base_Sleep_Disorder <- relevel(factor(Data.num$Sleep_Disorder), ref = "None")


filter <- sample(c(TRUE,FALSE), nrow(Data.num), replace=TRUE, prob=c(0.8,0.2))
train_dataset  <- Data.num[filter, ]
test_dataset  <- Data.num[!filter, ]  



class(train_dataset$Base_Sleep_Disorder )


target <- train_dataset$Base_Sleep_Disorder
features_train <- train_dataset[,c("Age", "Sleep_Duration", "Quality_of_Sleep", 
                                   "Physical_Activity_Level", "Stress_Level",
                                   "BMI_Category", "Blood_Pressure", 
                                   "Heart_Rate", "Daily_Steps", "Female", 
                                   "Male", "Accountant", "Doctor", "Engineer", 
                                   "Lawyer", "Manager", "Nurse",
                                   "Sales_Representative", "Salesperson", 
                                   "Scientist", "Software_Engineer", "Teacher")]

features_test <- test_dataset[,c("Age", "Sleep_Duration", "Quality_of_Sleep", 
                                 "Physical_Activity_Level", "Stress_Level",
                                 "BMI_Category", "Blood_Pressure", 
                                 "Heart_Rate", "Daily_Steps", "Female", 
                                 "Male", "Accountant", "Doctor", "Engineer", 
                                 "Lawyer", "Manager", "Nurse",
                                 "Sales_Representative", "Salesperson", 
                                 "Scientist", "Software_Engineer", "Teacher")]



# https://www.rdocumentation.org/packages/kernlab/versions/0.9-33/topics/ksvm
# https://uc-r.github.io/svm

# linear svm
svmfit1 <- svm(target~., data = features_train, kernel = "linear", scale = FALSE)
svmfit1

# with cross validation accuracy did not change
# svmfit1 <- svm(target~., data = features_train, kernel = "linear", cross=5, scale = FALSE)
# svmfit1

ypred <- predict(svmfit1, features_test)
(misclass <- table(predict = ypred, truth = test_dataset$Base_Sleep_Disorder))

confusionMatrix(misclass)



# SVM with kernel rbfdot (radial basis functi)
kernfit1 <- ksvm(target~.,data=features_train,kernel="rbfdot",
                 kpar=list(sigma=0.05),C=5,cross=3)
kernfit1

## predict mail type on the test set
ypred <- predict(kernfit1,features_test)

## Check results
(misclass <- table(predict = ypred, truth = test_dataset$Base_Sleep_Disorder))

confusionMatrix(misclass)


# SVM with kernel rbfdot: Create a kernel function using the build in rbfdot function
rbf <- rbfdot(sigma=0.01)
rbf

kernfit2 <- ksvm(target~.,data=features_train,type="C-bsvc",
                 kernel=rbf,C=10,prob.model=TRUE)



kernfit2

## Test on the training set with probabilities as output
ypred <- predict(kernfit2, features_test, type="probabilities")
ypred <- predict(kernfit2, features_test)

(misclass <- table(predict = ypred, truth = test_dataset$Base_Sleep_Disorder))

confusionMatrix(misclass)



#### Use custom kernel 

k <- function(x,y) {(sum(x*y) +1)*exp(-0.1*sum((x-y)^2))}
class(k) <- "kernel"


## train svm using custom kernel
kernfit3 <- ksvm(target~.,data=features_train,kernel=k,
                 C=5,cross=5)

kernfit3


## Test on the training set with probabilities as output
ypred <- predict(kernfit3, features_test)

(misclass <- table(predict = ypred, truth = test_dataset$Base_Sleep_Disorder))

confusionMatrix(misclass)

# https://stackoverflow.com/a/50427780/8185618
myKernels = c("vanilladot","polydot","besseldot")
results=list()
results_confusion_matrix=list()
for(i in 1:length(myKernels)){
  # call ksvm using  kernel instead of linear
  model <-  ksvm(as.matrix(features_train), target,type="C-svc",kernel=myKernels[[i]],C=100,scaled=TRUE) #as.factor(data[,11])
  # calculate a1.am
  a <- colSums(model@xmatrix[[1]] * model@coef[[1]])
  a
  # calculate a0
  a0 <- -model@b
  a0
  # see what the model predicts
  pred <- predict(model, features_test)
  pred
  # see what fraction of the model's predictions match the actual classification
  results[[i]]=data.table(kernel=myKernels[[i]],accuracy=sum(pred == test_dataset$Base_Sleep_Disorder) / nrow(test_dataset))
  accuracy=sum(pred == test_dataset$Base_Sleep_Disorder) / nrow(test_dataset)
  accuracy
  
  print(myKernels[[i]])
  (misclass <- table(predict = pred, truth = test_dataset$Base_Sleep_Disorder))
  print(confusionMatrix(misclass))
}
rbindlist(results)

# radial basis function kernel with these parameters
svmfit3 <- svm(target~., data = features_train, kernel = "radial", gamma = .1, cost = 10)
# plot classifier
svmfit3

## predict mail type on the test set
ypred <- predict(svmfit3, features_test)

## Check results
(misclass <- table(predict = ypred, truth = test_dataset$Base_Sleep_Disorder))

confusionMatrix(misclass)

# training linear kernel with different c's, to check accuracy 
possibleCosts <- c(0.001, 0.01, 0.1, 0.25, 0.5, 1,5,8,10, 12,100)

for(i in 1:length(possibleCosts))
{
  #svm.model <- svm(features_train, target, kernel="linear", cost=possibleCosts[i], type="C-classification")
  
  svm.model <- svm(target~., data = features_train, kernel = "linear", 
                   cost=possibleCosts[i], type="C-classification", scale = FALSE)
  print(possibleCosts[i])
  print(svm.model)
  
  ypred <- predict(svm.model, features_test)
  (misclass <- table(predict = ypred, truth = test_dataset$Base_Sleep_Disorder))
  
  print(confusionMatrix(misclass))
}