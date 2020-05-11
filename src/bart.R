#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#
#Author: Raquel AOki
#December 2019 
#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#

#Description: Given dataset, use BART and GFCI as causal methods 
rm(list=ls())


RUN_CATE = FALSE

#Save pred prob for py

#----------#----------#----------#----------#----------#----------#----------#
#BART 
#----------#----------#----------#----------#----------#----------#----------#
options(java.parameters = "-Xmx5g")
if(!require(bartMachine)) install.packages("bartMachine")
library(bartMachine)
#https://cran.r-project.org/web/packages/bartMachine/vignettes/bartMachine.pdf
#recommended package BayesTrees is not functional anymore 
  
setwd("~/GitHub/parkca")
filenames =  list.files(path = "GitHub/../data",all.files = TRUE)
i = 18
filenames[i]

data = read.table(paste('data/',filenames[i],sep=''), sep = ';', header = T)
data <- data[sample(nrow(data),replace=FALSE),] #shuffle data
 
#Splitting data
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
  
name = strsplit(filenames[i],"_", fixed= TRUE)
name = strsplit(name[[1]][length(name[[1]])],".", fixed = TRUE)[[1]][1]
name =  paste("./results/bart_model_",name,".rds", sep="")

#CHECK IF FILE EXIST
check =list.files(path = "GitHub/../results",all.files = TRUE)
if(sum( name == check) != 0){ 
	#load BART
	bart_machine <- readRDS(name)
}else{
	#BART model 
	bart_machine = bartMachine(X, y, num_trees = 50, num_burn_in = 500, num_iterations_after_burn_in = 1400 )
	summary(bart_machine)
	saveRDS(bart_machine, name)
} 
 
#making predictions
pred_p = predict(bart_machine, X_, type='prob') #returns the prob of being on label 1
predictions = data.frame(y_,pred_)
names(predictions) = c('obs','pred')
replace('.rds','.txt',name)
write.table(prdictions,gsub('.rds','.txt',name), sep = ';', row.names = FALSE)











  
  roc_data = data.frame(pred_p, y)
  names(roc_data) = c('pred','y01')
  #roc_data = rbind(def_roc(pred_p,y,0.01),def_roc(pred_p,y,0.02))
  #values = seq(0.03,1,by=0.01)
  #for(i in 1:length(values)){
  #  roc_data = rbind(roc_data,def_roc(pred_p,y,values[i]))
  #}
  #roc_data = data.frame(roc_data)
  #names(roc_data) = c('prob','tp1','fp1','tp2','fp2')
  write.table(roc_data,'results\\roc_bart_all.txt', row.names = FALSE,sep=';')
  
  #ROC CURVE PLOT
  #pred <- prediction(1-pred_p,y)
  #roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
  #plot(roc.perf, col = 'red',main='ROC')
  #abline(a=0, b= 1)
  #is 0.5 the best? 
  
  #def_threshold(y,pred_p,0.4)
  #def_threshold(y,pred_p,0.5)
  #def_threshold(y,pred_p,0.6)
  if(RUN_CATE){
  #making the interventional data, one for each gene 
  fit_test =  predict(bart_machine, data_test, type='prob')
  dif = data.frame(gene = names(data),mean=c(rep(999, dim(data)[2])), 
                   sd=c(rep(999, dim(data)[2])), se = c(rep(999, dim(data)[2])))
  #dif = read.table('results\\bart.txt', sep = ';', header=T)
  for(v in 1:dim(data_test)[2]){
    data_v = data_test
    data_v[,v] = 0
    fit = predict(bart_machine, data_v)
    dif$mean[v] = mean(fit_test-fit)
    dif$sd[v] = sd(fit_test-fit)
    dif$se[v] = mean((fit_test-fit)^2)
  }
  write.table(dif,'results\\feature_bart_all.txt', sep = ";", row.names = FALSE)
  }
}



if(RUN_BART&RUN_){
  options(java.parameters = "-Xmx5g")
  library(bartMachine)
  library(ROCR)
  set_bart_machine_num_cores(4) #new
  #https://cran.r-project.org/web/packages/bartMachine/vignettes/bartMachine.pdf
  #recommended package BayesTrees is not functional anymore 
  
  setwd("~/GitHub/project_spring2019")
  files = read.table('data/files_names.txt', sep = ';', header = T)
  files$files = paste('data/',files$files, sep='')
  files$ci = as.character(files$ci)
  files$class = as.character(files$class)
  
  for(f in 1:dim(files)[1]){
    data = read.table(files$files[f], sep = ';', header = T)
    if(dim(data)[1]>=100){
    data <- data[sample(nrow(data),replace=FALSE),]
    #testing set
    extra_test = data[1:round(dim(data)*0.3)[1],c(1,2,3)]
    data_test = data[1:round(dim(data)*0.3)[1],-c(1,2,3)]
    y_test = as.factor(extra_test$y)
    #training set 
    extra =  data[-c(1:round(dim(data)*0.3)[1]),c(1,2,3)]
    data = data[-c(1:round(dim(data)*0.3)[1]),-c(1,2,3)]
    y = as.factor(extra$y)
    
    #fitting the BART model 
    bart_machine = bartMachine(data, y, num_trees = 50, num_burn_in = 500, num_iterations_after_burn_in = 1400 )
    #summary(bart_machine)
    
    #checking BART convergence 
    #plot_convergence_diagnostics(bart_machine)
    
    #making predictions
    pred_p = predict(bart_machine, data, type='prob') #returns the prob of being on label 1
    
    
#    roc_data = rbind(def_roc(pred_p,y,0.01),def_roc(pred_p,y,0.02))
#    values = seq(0.03,1,by=0.01)
#    for(i in 1:length(values)){
#      roc_data = rbind(roc_data,def_roc(pred_p,y,values[i]))
#    }
#    roc_data = data.frame(roc_data)
#    names(roc_data) = c('prob','tp1','fp1','tp2','fp2')
    
    roc_data = data.frame(pred_p, y)
    names(roc_data) = c('pred','y01')   
    write.table(roc_data,paste('results\\roc_bart_',files$ci[f],'_',files$class[f],'.txt',sep=''), row.names = FALSE,sep=';')
    
    #ROC CURVE PLOT
    #pred <- prediction(1-pred_p,y)
    #roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
    #plot(roc.perf, col = 'red',main='ROC')
    #abline(a=0, b= 1)
    #is 0.5 the best? 
    
    #def_threshold(y,pred_p,0.4)
    #def_threshold(y,pred_p,0.5)
    #def_threshold(y,pred_p,0.6)
    
    #making the interventional data, one for each gene 
    if(RUN_CATE){
      
    fit_test =  predict(bart_machine, data_test, type='prob')
    dif = data.frame(gene = names(data),mean=c(rep(999, dim(data)[2])), 
                     sd=c(rep(999, dim(data)[2])), se = c(rep(999, dim(data)[2])))
    #dif = read.table('results\\bart.txt', sep = ';', header=T)
    for(v in 1:dim(data_test)[2]){
      data_v = data_test
      data_v[,v] = 0
      fit = predict(bart_machine, data_v)
      dif$mean[v] = mean(fit_test-fit)
      dif$sd[v] = sd(fit_test-fit)
      dif$se[v] = mean((fit_test-fit)^2)
    }
    write.table(dif,paste('results\\feature_bart_',files$ci[f],'_',files$class[f],'.txt',sep=''), sep = ";", row.names = FALSE)
    }
    
  }
}
}

#----------#----------#----------#----------#----------#----------#----------#
#GFCI - works on my laptop only because the java dependences
#----------#----------#----------#----------#----------#----------#----------#

if(RUN_FCI){

  Sys.setenv(JAVA_HOME='C:/Program Files (x86)/Java/jre1.8.0_231')
  Sys.setenv(JAVA_HOME='C:/Program Files/Java/jre1.8.0_231')
  
  require(rJava)
  require(stringr)
  library(devtools)
  library(rcausal)

  
  ########################## 
  #Toy example
  ########################## 
  
  #data("charity")   #Load the charity dataset
  
  #tetradrunner.getAlgorithmDescription(algoId = 'fges')
  #tetradrunner.getAlgorithmParameters(algoId = 'fges',scoreId = 'fisher-z')
  #Compute FGES search
  #tetradrunner <- tetradrunner(algoId = 'fges',df = charity,scoreId = 'fisher-z',
  #                             dataType = 'continuous',alpha=0.1,faithfulnessAssumed=TRUE,maxDegree=-1,verbose=TRUE)
  
  #tetradrunner$nodes #Show the result's nodes
  #tetradrunner$edges #Show the result's edges
  
  #Source
  #https://bd2kccd.github.io/docs/r-causal/
  #https://arxiv.org/ftp/arxiv/papers/1507/1507.07749.pdf
  ########################## 
  #Real Code
  #Raquel AOki
  ########################## 
  
  setwd("~/GitHub/project_spring2019")
  data = read.table('data/tcga_train_gexpression_cgc_7k.txt', sep = ';', header = T)
  data <- data[sample(nrow(data),replace=FALSE),]
  #testing set
  extra_test = data[1:round(dim(data)*0.3)[1],c(1,2,3)]
  data_test = data[1:round(dim(data)*0.3)[1],-c(1,2,3)]
  y_test = as.factor(extra_test$y)
  #training set 
  extra =  data[-c(1:round(dim(data)*0.3)[1]),c(1,2,3)]
  data = data[-c(1:round(dim(data)*0.3)[1]),-c(1,2,3)]
  y = as.factor(extra$y)

  
  #t = c(3000)
  #test = c()
  #for(t0 in 1:length(t)){
    bd = data[,1:1000]
    #tetradrunner.getAlgorithmDescription(algoId = 'gfci ')
    #tetradrunner.getAlgorithmParameters(algoId = 'gfci',scoreId = 'fisher-z', testID = "correlation-t")
    #Compute FGES search
    tetradrunner <- tetradrunner(algoId = 'gfci',df = bd,scoreId = 'fisher-z',
                                 testID = 'fisher-z',
                                 dataType = 'continuous',alpha=0.05,
                                 faithfulnessAssumed=TRUE,maxDegree=30,verbose=TRUE)
    length(tetradrunner$edges)
    #testID tetradrunner.listIndTests()
    #algoID = tetradrunner.listAlgorithms()
    #scoreId tetradrunner.listScores()
    
   # fgs <- fgs(df = charity, penaltydiscount = 2, depth = -1, ignoreLinearDependence = TRUE, 
    #           heuristicSpeedup = TRUE, numOfThreads = 2, verbose = TRUE, priorKnowledge = prior)
    #head(tetradrunner$nodes) #Show the result's nodes
    #head(tetradrunner$edges) #Show the result's edges
    #test[t0] = length(tetradrunner$edges)
  #}
  
  df = data.frame(tetradrunner$edges)
  
  df[,1] = as.character(df[,1])
  
  
  
  write.table(df, file = 'results/example_edges.txt', row.names = FALSE, sep = ';')
  
  #plot
  library(DOT)
  graph_dot <- tetradrunner.tetradGraphToDot(tetradrunner$graph)
  dot(graph_dot)
  
  
  #trying pcalc again 
  #https://cran.r-project.org/web/packages/pcalg/vignettes/pcalgDoc.pdf
  #https://cran.r-project.org/web/packages/pcalg/pcalg.pdf
  library(pcalg)
  
  setwd("~/GitHub/project_spring2019")
  data = read.table('data/tcga_train_gexpression_cgc_7k.txt', sep = ';', header = T)
  data <- data[sample(nrow(data),replace=FALSE),]
  #testing set
  extra_test = data[1:round(dim(data)*0.3)[1],c(1,2,3)]
  data_test = data[1:round(dim(data)*0.3)[1],-c(1,2,3)]
  y_test = as.factor(extra_test$y)
  #training set 
  extra =  data[-c(1:round(dim(data)*0.3)[1]),c(1,2,3)]
  data = data[-c(1:round(dim(data)*0.3)[1]),-c(1,2,3)]
  y = as.factor(extra$y)
  
  int = seq(80,250,by=10)
  times = c()
  for(i in 1:length(int)){
    ptm <- proc.time()
    bd = data[,c(1:int[i])]
    suffStat = list(C = cor(bd), n = nrow(bd))
    model = rfci(suffStat, indepTest = gaussCItest, alpha = 0.01, m.max = 10, numCores = 3, skel.method = "stable.fast",
                 conservative = FALSE, labels = names(bd))  
    aux = proc.time() - ptm
    times = c(times, aux[1])
  }
  
}