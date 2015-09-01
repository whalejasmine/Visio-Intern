setwd("c:/Users/jasmine.qi/Documents/data cleaning")

library(RSNNS)

library(caret)
library(nnet)

write.csv(total,"datawocolin.csv")

total= read.csv("deriveclean.csv")
total$Units=as.factor(total$Units)
#encoding  N = -1 Y = 1  
contrasts(total$Borrower.Age.Group)=contr.treatment(6)
contrasts(total$Size)=contr.treatment(6)
contrasts(total$Loan.Type)=contr.treatment(2)
contrasts(total$Loan.Term)=contr.treatment(6)
contrasts(total$Loan.Purpose)=contr.treatment(2)
contrasts(total$Occupant)=contr.treatment(3)
contrasts(total$Assignment.Type)=contr.treatment(3)
contrasts(total$Location)=contr.treatment(3)
contrasts(total$Built.up)=contr.treatment(3)
contrasts(total$Subject..PropertyState)=contr.treatment(30)
contrasts(total$Male.Female)=contr.treatment(2)
contrasts(total$Growth)=contr.treatment(3)
contrasts(total$Property.Values)=contr.treatment(3)
contrasts(total$Demand.Supply)= contr.treatment(3)
contrasts(total$Marketing.Time)=contr.treatment(3)
contrasts(total$Units)=contr.treatment(4)
contrasts(total$Type)=contr.treatment(3)
contrasts(total$Condition)=contr.treatment(6)
contrasts(total$Rent.Own)= contr.treatment(2)
contrasts(total$Full.or.Part.Time.investor)=contr.treatment(2)

total$Borrower.Age.Group=model.matrix(~factor(total$Borrower.Age.Group))
total$Size=model.matrix(~factor(total$Size))
total$Loan.Type=model.matrix(~factor(total$Loan.Type))
total$Loan.Term=model.matrix(~factor(total$Loan.Term))
total$Loan.Purpose=model.matrix(~factor(total$Loan.Purpose))
total$Occupant=model.matrix(~factor(total$Occupant))
total$Assignment.Type=model.matrix(~factor(total$Assignment.Type))
total$Location=model.matrix(~factor(total$Location))
total$Built.up=model.matrix(~factor(total$Built.up))
total$Subject..PropertyState=model.matrix(~factor(total$Subject..PropertyState))
total$Male.Female=model.matrix(~factor(total$Male.Female))
total$Growth=model.matrix(~factor(total$Growth))
total$Property.Values=model.matrix(~factor(total$Property.Values))
total$Demand.Supply= model.matrix(~factor(total$Demand.Supply))
total$Marketing.Time=model.matrix(~factor(total$Marketing.Time))
total$Units=model.matrix(~factor(total$Units))
total$Type=model.matrix(~factor(total$Type))
total$Condition=model.matrix(~factor(total$Condition))
total$Rent.Own= model.matrix(~factor(total$Rent.Own))
total$Full.or.Part.Time.investor=model.matrix(~factor(total$Full.or.Part.Time.investor))



total$LoanID=NULL
total$X = NULL
dim(total)
total=total[,-corr]


##preprocess transform
nums = total[, sapply(total, is.numeric)] #subset numeric varables
categ = total[, sapply(total, is.factor)]

#trans.box = preProcess(nums, method =c("BoxCox", "scale", "center") )
#transformed.box = predict(trans.box, nums)
#write.csv(transformed.box,"transformednumbox.csv")
#write.csv(categ, "cate.csv")
#num=read.csv("transformednumbox.csv")
#cate= read.csv("cate.csv")
#total=merge(num, cate, by="X")
#write.csv(total, "datanums.csv")
total$X=NULL


totalTrans=preProcess(total, method = "BoxCox")
total.box=predict(totalTrans, total)



indexes = sample(1:nrow(total), size= 0.7*nrow(total))

train = total[indexes, ]
dim(train)
myvars =names(train) %in% "Loanscore"
trainX= train[!myvars]
trainY =train$Loanscore#/( range(train$Loanscore)[2]-range(train$Loanscore)[1])
  #trainY= log(train$Loanscore)          

test = total[-indexes, ]
myvarsTest = names(test) %in% "Loanscore"
testX= test[!myvarsTest]
testY = test$Loanscore
  #normalizeData(test$Loanscore)

## train lm

lm.train= train(trainX, trainY, method="lm", trainControl=trainControl(method="cv"))

## Multiplayer 
mlptrain= mlp(trainX, trainY, size=c(5), learnFuncParams = c(0.1), maxit = 100,
               inputsTest = testX, targetsTest = testY, metrix="RSME", linOut = F)



##NW

library(neuralnet)

nnetFit <- nnet(trainX, trainY,
                size = 5,
                decay = 0.01,
               linout = TRUE,
                ## Reduce the amount of printed output
                   trace = FALSE,
                 ## Expand the number of iterations to find
                   ## parameter estimates..
                   maxit = 500,
                 ## and the number of parameters used by the model
                  MaxNWts = 5 * (ncol(trainX) + 1) + 5 + 1)

fit.nw = 
tooHigh = findCorrelation(cor(train, use="complete.obs"), cutoff=0.75)
trainXnnet =trainX[, -tooHigh]
trestXnnet = testX[, -tooHigh]

nnetGrid = expand.grid(.decay = c(0, 0.01, .1),
                         .size = c(1:10),
                         .bag = FALSE)

nnetTune = train(trainX, trainY,
                  method = "nnet",
                #tuneGrid = nnetGrid,
                trainControl=trainControl(method="cv"),
                  #preProc = c("center", "scale"),
                  linout = TRUE,
                  trace = FALSE)#MaxNWts = 10 * (ncol(trainXnnet) + 1) + 10 + 1)

##MARS 
library(MARSS)
library(earth)
fit = earth(trainX, trainY)
fit.pred= predict(fit)              

fit.test=predict(fit, test)
fittest.r2 = cor(fit.test,test$Loanscore, method = "spearman")
plot(test$Loanscore, fit.test, col=c("red", "blue"))

marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)

# Fix the seed so that the results can be reproduced
 marsTuned <- train(trainX, trainY,
                      method = "earth",
                     # Explicitly declare the candidate models to test
                       tuneGrid = marsGrid,
                      trControl = trainControl(method = "cv"))
         
mars.predict = predict(marsTuned)
mar.r = R2(train$Loanscore, mars.predict)
plot(train$Loanscore, mars.predict, col=c("red", "blue"))



##kNN
# Remove a few sparse and unbalanced fingerprints first
knnDescr <- trainX[, -nearZeroVar(trainX)] 
 knnTune <- train(knnDescr,
                   trainY,
                    method = "knn",
                 # preProc=c("center", "scale"),
                    tuneGrid = data.frame(.k = 1:20),
                   trControl = trainControl(method = "cv"))
 
 
 ##svm
 library(e1071)
 
 svmRTuned <- train(trainX, trainY,
                     method = "svmRadial",
                    tuneLength = 14,
                     trControl = trainControl(method = "cv"))
