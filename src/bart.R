#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#
#Author: Raquel AOki
#December 2019
#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#

#Description: Given dataset, use BART and GFCI as causal methods
rm(list=ls())


RUN_CATE = TRUE

#Save pred prob for py

#----------#----------#----------#----------#----------#----------#----------#
#BART
#----------#----------#----------#----------#----------#----------#----------#
options(java.parameters = "-Xmx5g")
#if(!require(bartMachine)) install.packages("bartMachine")
library(bartMachine)
library(factoextra)

set.seed(99999)

#https://cran.r-project.org/web/packages/bartMachine/vignettes/bartMachine.pdf
#recommended package BayesTrees is not functional anymore

setwd("~/GitHub/parkca")
filenames =  list.files(path = "GitHub/../data",all.files = TRUE)
filenames = filenames[filenames!='.']
filenames = filenames[filenames!='..']
filenames = filenames[16:18]


for(i in 1:length(filenames)){

cat('\n')
cat(filenames[i])
data = read.table(paste('data/',filenames[i],sep=''), sep = ';', header = T)
data <- data[sample(nrow(data),replace=FALSE),] #shuffle data


if(dim(data)[1]>800){

#Splitting data
set.seed(1+i)
train_index <- sample(1:nrow(data), 0.8 * nrow(data))
test_index <- setdiff(1:nrow(data), train_index)

#Training
E = data[train_index,c(1,2,3)] #Extra
X = data[train_index,-c(1,2,3)]
y = as.factor(E$y)

#Testing
E_ = data[test_index,c(1,2,3)]
X_ = data[test_index,-c(1,2,3)]
y_ = as.factor(E_$y)

#Latent features
pca = prcomp(X, center = TRUE,scale. = TRUE)
Xa= predict(pca, newdata = X)[,c(1:30)]
Xa_= predict(pca, newdata = X_)[,c(1:30)]

#Names
name = strsplit(filenames[i],"_", fixed= TRUE)
name = strsplit(name[[1]][length(name[[1]])],".", fixed = TRUE)[[1]][1]
coef_name = paste('bart_', name, sep = '')
name =  paste("./results/bart_",name,".rds", sep="")

#CHECK IF FILE EXIST
check =list.files(path = "GitHub/../results",all.files = TRUE)
check = paste('./results/',check,sep='')
if(sum( name == check) != 0){
  #load BART
  bart_machine <- readRDS(name)
}else{
  #BART model
  bart_machine = bartMachine(data.frame(X,Xa), y, num_trees = 50, num_burn_in = 500, num_iterations_after_burn_in = 1400,serialize = TRUE)
  summary(bart_machine)
  #saveRDS(bart_machine, name)
}

#making predictions
pred_ = predict(bart_machine, data.frame(X_,Xa_), type='prob') #'class' or 'prob'
predictions = data.frame(y_,pred_)
names(predictions) = c('obs','pred')
write.table(predictions,gsub('.rds','.txt',name), sep = ';', row.names = FALSE)



if(RUN_CATE){
  #making the interventional data, one for each gene
  pred_ =  mean(predict(bart_machine, data.frame(X_,Xa_), type='prob'))
  coef = data.frame(gene = names(X), cate = c(rep(999, dim(X)[2])))

  if(i==1){
    coef_ = data.frame(gene = names(X))
    coef_$current = coef$bart_MALE
  }else{
    coef_$current = c(999)
  }

  #Simulations
  s = 60
  for(v in 7001:dim(X_)[2]){
    X_i = X_
    X_i[,v] = 0

    pred_i = c(rep(0,s))
    pred_i_all = predict(bart_machine, data.frame(X_i,Xa_))
    set.seed(10+i)
    for(s0 in 1:s){
      #X_i_s = X_i[sample(dim(X_i)[1],floor(dim(X_i)[1]*0.85)),]
      pred_i[s0] = mean(sample(pred_i_all,floor(dim(X_i)[1]*0.85)))
    }
    dif = pred_ - pred_i
    dif0 = mean(dif)/(var(dif)/s)^0.5
    if(dif0>qnorm(0.025) & dif0<qnorm(0.975)) coef_$current[v] =  mean(dif) else coef_$current[v] =  0

    #Saving
    if(v%%200==0){
      cat(v)
      cat(' of ')
      cat(dim(X_)[2])
      cat('\n')
      coef$cate = coef_$current
      write.table(coef,'results\\coef_bart.txt', sep = ";", row.names = FALSE)
      saveRDS(coef_, coef_name)
    }

  }
  
  coef[,dim(coef)[2]]=coef_$current
  names(coef)[dim(coef)[2]] = coef_name
  coef_$current = c(999)
  
  write.table(coef,'results\\coef_bart.txt', sep = ";", row.names = FALSE)
  saveRDS(coef_, coef_name)
}

}

}

if(RUN_CATE){
  coef_ = subset(coef_, select = -c(current))
  write.table(coef_,'results\\coef_bart.txt', sep = ";", row.names = FALSE)
  saveRDS(coef_, coef_name)
}


for(i in 1:length(filenames)){
  
  data = read.table(paste('data/',filenames[i],sep=''), sep = ';', header = T)
  data <- data[sample(nrow(data),replace=FALSE),] #shuffle data
  cat('\n')
  cat(filenames[i])
  cat(dim(data)) 
  
  }

