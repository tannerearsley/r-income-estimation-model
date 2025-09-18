# Adult Income Prediction Modeling
# Author: Tanner Earsley
# Created: 9/14/2025
# Last Updated: 9/17/2025

# Project Overview:
# This project uses the UCI Adult Income dataset to predict whether an individual's income exceeds $50,000 annually. 
#Sourced from the 1994 U.S. Census Bureau (Current Population Survey)

# Methodology:
# Predictive supervised models such as SVM, Regression, and Decision Trees will be applied to the dataset.

# Evaluation:
# Model predictions will be compared to the actual income labels to assess accuracy.
# We will then determine which model is the best fit

###Initialize packages and data
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("e1071")
library(e1071)
library(rpart)
library(rpart.plot)
#import all libraries to be used in this project
set.seed(57) #set seed for constant random value output on each session

incomedata <- read.table("adult.data", sep=",",stringsAsFactors=TRUE,header=FALSE, na.strings=" ?")
#Import data, convert strings to factors, and set all " ?" values to NA (this is what the data set uses as Null)
colnames(incomedata) <- c("Age","WorkClass","FnlWgt", "Education", "Education_Num","Marital_Status",
                          "Occupation", "Relationship","Race","Sex", "Capital_Gain","Capital_Loss",
                          "Hours_Week", "Native_Country","Income") #add column names

#lapply(incomedata, function(x) if (is.factor(x)) levels(x))
#/\ Running the above will list all levels in the dataset, important for knowing data range/options of input

summary(incomedata)
head(incomedata)
tail(incomedata) #Lets get a working look at the data
levels(incomedata$Income)

###Data Cleaning
colSums(is.na(incomedata))#How many na's in each column?
perc <- sum(!complete.cases(incomedata))/nrow(incomedata) # what % contain na data?
print(paste0("Rows containing nulls: ", perc, " >> ", round(perc*100, 2), "%"))
nrow(incomedata)#Willing to accept dropping 7% of data cases, as there are ample row entries ()
incomedata <- na.omit(incomedata)#Remove all rows with "na" data
colSums(is.na(incomedata))
nrow(incomedata)#"Only" 2,399 rows lost

incomedata <- subset(incomedata, select = -c(FnlWgt, Education_Num))#Remove columns with no impact or repeats (Education vs Ed_Num)


#Separate data into training and test data sets. 80% training and 20% test is the industry standard
train_rows <- sample(1:nrow(incomedata), size = 0.8*nrow(incomedata))
train_data <- incomedata[train_rows,]
test_data  <- incomedata[-train_rows,]

##SVM Model
#find hyperplane that best separates two classes of income, with error margin inclusion
income_svm <- svm(Income~.,data=train_data, scale=TRUE, kernel="linear")
pred_svm <- predict(income_svm, newdata = test_data)#predict Income level on test data set
svm_outcome <- table(pred_svm, test_data$Income) # create a confusion matrix. True Positive, False Positive, etc...
rownames(svm_outcome) <- c("  <=50k: Est", "   >50k: Est")#For easier reading
colnames(svm_outcome) <- c("  <=50k: Act", "  >50k: Act")

svm_outcome
svm_accuracy <- sum(diag(svm_outcome)) / sum(svm_outcome)#Calculate the accuracy of the model
cat("Accuracy: ",svm_accuracy," >> ", round(svm_accuracy*100, 2), "%", sep="")

##Regression Model
income_regr <- glm(Income~.,data=train_data, family=binomial) #Establish a baseline value and compare the rest against it
#Important: We use logistic regression (response=categorical), instead of normal regression (response=numeric/continuous)
summary(income_regr)#Show log odds and their effects.
exp(coef(income_regr))#Exponentiates, turning log odds into actual ratios
#Notable observations:
#Male has 2.39 times the odds of baseline Female value
#Race Asian-Pac-Islander has 2.31 times observed odds of baseline value Amer-Indian-Eskimo
#Despite being the baseline, the private sector has the greatest influence. Other sectors reduce chances
#As education increases, chance of making above 50k steadily increases each level
pred_regr <- predict(income_regr, newdata = test_data, type="response")#predict Income level on test data set
regr_class <- ifelse(pred_regr > 0.5, ">50K", "<=50K")
regr_outcome <- table(regr_class, test_data$Income) # create a confusion matrix. True Positive, False Positive, etc...
rownames(regr_outcome)<-c("  <=50k: Est","   >50k: Est")
colnames(regr_outcome)<-c("  <=50k: Act","  >50k: Act")

regr_outcome
regr_accuracy <- sum(diag(regr_outcome)) / sum(regr_outcome)#Calculate the accuracy of the model
cat("Accuracy: ",regr_accuracy," >> ", round(regr_accuracy*100, 2), "%", sep="")

#Decision Tree Model
#Choosing 0.05 as a middle value between 0.01 (too simple) and 0.001 (hard to read and possibly overfit)
#Choose type 1 to see nodes at each point, and #104 for percentage plus classification (?rpart.plot, 100+4)
#The darker blue, the more likely base value <=50k is true. The greener, the more likely >50k is the case.
income_dtree<-rpart(Income~.,data=train_data,method="class",control=rpart.control(cp=0.005))#recursive partition to build the tree
rpart.plot(income_dtree,type=1,extra=104,fallen.leaves=TRUE,main="Income Prediction Decision Tree")
pred_dtree<-predict(income_dtree,newdata=test_data,type="class")
dtree_outcome<-table(pred_dtree,test_data$Income)
rownames(dtree_outcome)<-c("  <=50k: Est","   >50k: Est")
colnames(dtree_outcome)<-c("  <=50k: Act","  >50k: Act")
#My favorite one! Actually ends up having the best accuracy
dtree_outcome
dtree_accuracy<-sum(diag(dtree_outcome))/sum(dtree_outcome)
cat("Accuracy: ",dtree_accuracy," >> ",round(dtree_accuracy*100,2),"%",sep="")


result_acc <- data.frame(#Create a table of all model accuracies
  Model = c("SVM", "Logistic Regression", "Decision Tree"),
  Accuracy = c(svm_accuracy, regr_accuracy, dtree_accuracy)
)
print(result_acc)#Decision tree wins

#Manual Input Prediction
user_profile <- data.frame(
  Age = 22,
  WorkClass = factor(" Private", levels = levels(incomedata$WorkClass)),
  Education = factor(" Bachelors", levels = levels(incomedata$Education)),#Masters is not complete yet
  Marital_Status = factor(" Never-married", levels = levels(incomedata$Marital_Status)),
  Occupation = factor(" Tech-support", levels = levels(incomedata$Occupation)),
  Relationship = factor(" Not-in-family", levels = levels(incomedata$Relationship)),
  Race = factor(" White", levels = levels(incomedata$Race)),
  Sex = factor(" Male", levels = levels(incomedata$Sex)),
  Capital_Gain = 1000,
  Capital_Loss = 4400,
  Hours_Week = 45,
  Native_Country = factor(" United-States", levels = levels(incomedata$Native_Country))
)

#Predict using SVM
svm_Tpredict <- predict(income_svm, user_profile)
as.numeric(predict(income_svm, user_profile))#Raw numeric of 2c
cat("SVM Prediction:", as.character(svm_Tpredict), "\n")

#Predict using Logistic Regression
regr_Tpredict <- predict(income_regr, user_profile, type = "response")
regr_Tclass <- ifelse(regr_Tpredict > 0.5, ">50K", "<=50K")
predict(income_regr, user_profile, type="response")#Log Reg 0.78 >> 1, close to 0 is <=50, vice versa for 1
cat("Logistic Regression Prediction:", regr_Tclass, "\n")

#Predict using Decision Tree
dtree_Tpredict <- predict(income_dtree, user_profile, type="class")
as.numeric(predict(income_dtree, user_profile, type="class"))#Dtree numeric of 1, 1(Base) <=50k, 2  is >50k
cat("Decision Tree Prediction:", as.character(dtree_Tpredict), "\n")

user_pred <- data.frame(
  Model = c("SVM", "Logistic Regression", "Decision Tree"),
  User_Est = c(
    as.character(svm_Tpredict),
    regr_Tclass,
    as.character(dtree_Tpredict)
  )
)
print(user_pred)
#Expected variance in models, as at the time of coding this, my income is $59280 with 45 hours & overtime (pretax)
#Code can be altered with different input to further test model



