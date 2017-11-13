library(caret)
library(dplyr)
library(rattle)
rm(list=ls())
dir<-getSrcDirectory(function(x) {x})
setwd(dir)
fileurl1<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileurl1,destfile="dataset_train_8.csv")
data_train<-read.csv("dataset_train_8.csv")
fileurl2<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileurl2,destfile="dataset_test_8.csv")
data_test<-read.csv("dataset_test_8.csv")
##exploaratory data analysis
str(data_train)##echo off, (160 variables) we have NA in data
table(data_train$classe)
na_indices<-which(colSums(is.na(data_train))>=10000)#feature dimensionality reduction
data_train=data_train[,-na_indices]
data_test=data_test[,-na_indices]
data_train=data_train[,-c(1:5)]#contain name, index, and time stamp
data_test=data_test[,-c(1:5)]
table(is.na(data_train))
nzv_index<-nearZeroVar(data_train)
data_train<-data_train[,-nzv_index]
data_test<-data_test[,-nzv_index]
inTrain<-createDataPartition(y=data_train$classe,p=0.7,list=FALSE)
train_set<-data_train[inTrain,]
valid_set<-data_train[-inTrain,]


##Fitting models
##with cross validation applied  
##why random forest? more acurate one of the most popular models used in machine learning
#more computationally intensive
##fit classe vs others using two different models rf and rpart(decision tree) and 
##compare confusion matrix 
##set seed
set.seed(111)
a<-which(names(data_train)=='classe')
pca_preproc<-preProcess(train_set[,-a],method="pca",thres=.9)
train_pc<-predict(pca_preproc,train_set[,-a])#have to do the same to test and validate
valid_pc<-predict(pca_preproc,valid_set[,-a])
test_pc<-predict(pca_preproc,data_test[,-a])
train_control<-trainControl('cv',10)
system.time(tree_model<-train(y=train_set$classe,x=train_pc,trControl=train_control,method="rpart"))
fancyRpartPlot(tree_model$finalModel)
predict_tree<-predict(tree_model,newdata=valid_pc)
cf_rpart<-confusionMatrix(predict_tree,valid_set$classe)
cf_rpart
cf_rpart<-cf_rpart$overall['Accuracy']
system.time(rf_model<-train(y=train_set$classe,x=train_pc,trControl=train_control,method="rf"))
predict_rf<-predict(rf_model,newdata=valid_pc)
cf_rf<-confusionMatrix(predict_rf,valid_set$classe)
cf_rf
cf_rf_acc<-cf_rf$overall['Accuracy']
predict_test<-predict(rf_model,newdata=test_pc)
out<-data.frame(test=1:20,prediction=predict_test)
write.table(out,"output.txt",sep="\t",row.names=FALSE)  
