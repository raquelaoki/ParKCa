rm(list=ls())
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

setwd("~\\Documents\\GitHub\\ParKCa")

#------------------------- CHANGE HERE TO DOWNLOAD DATA AGAIN
donwload_clinical = FALSE
process_clinical = FALSE
genes_selection = FALSE


theRootDir <- "~\\Documents\\GitHub\\ParKCa\\data\\"
diseaseAbbrvs <- c("ACC", "BLCA", "BRCA", "CHOL", "ESCA", "HNSC", "LGG", "LIHC", "LUSC", "PAAD", "PRAD", "SARC", "SKCM", "TGCT", "UCS")
diseaseAbbrvs_l <- c("acc", 'BRCA' ,"blca", "chol","esca", "hnsc", "lgg", "lihc", "lusc",  "paad", "prad", "sarc", "skcm",  "tgct", "ucs")


#------------------------ DOWNLOAD CLINICAL INFORMATION
#Downloading the clinical data using https://raw.github.com/paulgeeleher repo
if(donwload_clinical){
  clinicalFilesDir <- paste(theRootDir, "clinical/", sep="")
  dir.create(clinicalFilesDir, showWarnings = FALSE)

  for(i in 1:length(diseaseAbbrvs)){
    fname <- paste("nationwidechildrens.org_clinical_patient_", allTcgaClinAbrvs[i], ".txt", sep="")
    theUrl <- paste("https://raw.github.com/paulgeeleher/tcgaData/master/nationwidechildrens.org_clinical_patient_", allTcgaClinAbrvs[i], ".txt", sep="")
    download.file(theUrl, paste(clinicalFilesDir, fname, sep=""))
  }
}


#------------------------ CLINICAL INFORMATION DATA PROCESSING
#NOTE: columns have different names in each cancer type. These are more commom among them all
if(process_clinical){
  cnames = c("bcr_patient_barcode",'new_tumor_event_dx_indicator','abr')


  #Files names
  fname1 <- paste(clinicalFilesDir,"nationwidechildrens.org_clinical_patient_",diseaseAbbrvs_l,".txt" , sep='')

  #Rotine to read the files, select the important features, and bind in a unique dataset
  i = 1
  bd.aux = read.csv(fname1[i], sep = "\t")
  bd.aux$abr = diseaseAbbrvs[i]
  bd.c = subset(bd.aux, select = cnames)

  for(i in 2:length(fname1)){
    bd.aux = read.csv(fname1[i], sep = "\t", header = T)
    bd.aux$abr = diseaseAbbrvs[i]
    bd.c = rbind(bd.c, subset(bd.aux, select = cnames))
  }

  bd.c = subset(bd.c, new_tumor_event_dx_indicator=="YES"|new_tumor_event_dx_indicator=="NO")
  bd.c$new_tumor_event_dx_indicator  = as.character(bd.c$new_tumor_event_dx_indicator)

  write.table(bd.c,paste(theRootDir,'tcga_cli.txt',sep=''), row.names = F, sep = ';')
}


#-------------------------- GENE EXPRESSION GENE SELECTION - keeping the driver genes


if(genes_selection){
  bd = read.table(paste(theRootDir,'tcga_rna_old.txt',sep=''), header=T, sep = ';')
  bd = subset(bd, select = -c(patients2))
  head(bd[,1:10])
  dim(bd)

  cl = read.table(paste(theRootDir,'tcga_cli_old.txt',sep=''), header=T, sep = ';')
  cl = subset(cl, select = c(patients, new_tumor_event_dx_indicator,abr))
  names(cl)[2] = 'y'
  cl$y = as.character(cl$y)
  cl$y[cl$y=='NO'] = 0
  cl$y[cl$y=='YES'] = 1

  bd1 = merge(cl,bd,by.x = 'patients',by.y = 'patients', all = F)
  head(bd1[,1:10])

  cgc = read.table(paste(theRootDir,'cancer_gene_census.csv',sep = ''),header=T, sep=',')[,c(1,5)]

  #eliminate the ones with low variance
  require(resample)
  exception = c(1,2,3)
  var = colVars(bd1[,-exception])
  var[is.na(var)]=0
  datavar = data.frame(col = 1:dim(bd1)[2], colname = names(bd1), var = c(rep(100000,length(exception)),var))

  #adding driver gene info
  #42 are not found
  datavar = merge(datavar, cgc, by.x='colname','Gene.Symbol',all.x=T)
  rows_eliminate = rownames(datavar)[datavar$var<500 & is.na(datavar$Tier)]#26604.77
  datavar = datavar[-as.numeric(as.character(rows_eliminate)),]

  bd1 = bd1[,c(datavar$col)]
  order = c('patients','y','abr',names(bd1))
  order = unique(order)
  bd1 = bd1[,order]
  head(bd1[,1:10])

  #eliminate the ones with values between 0 and 1 are not signnificantly different
  bdy0 = subset(bd1, y==0)
  bdy1 = subset(bd1, y==1)
  pvalues = rep(0,dim(bd1)[2])
  pvalues_ks = rep(0,dim(bd1)[2])
  for(i in (length(exception)+1):dim(bd1)[2]){
    #pvalues[i] =  t.test(bdy0[,i],bdy1[,i])$p.value
    bd1[,i] = log(bd1[,i]+1)
    pvalues[i] = wilcox.test(bdy0[,i],bdy1[,i])$p.value
    #pvalues_ks[i] = ks.test(bdy0[,i],bdy1[,i])$p.value
  }


  #t.test:
  #H0: y = x
  #H1: y dif x
  #to reject the null H0 the pvalue must be <0.5
  #i want to keep on my data the genes with y dif x/small p values.
  datap = data.frame(col = 1:dim(bd1)[2], colname = names(bd1), pvalues = pvalues)
  datap = merge(datap, cgc, by.x='colname','Gene.Symbol',all.x=T)
  rows_eliminate =    rownames(datap)[datap$pvalues   >0.01 & is.na(datap$Tier)]
  #rows_eliminate_ks = rownames(datap)[datap$pvalues_ks>0.01 & is.na(datap$Tier)]
  #rows_eliminate = unique(rows_eliminate,rows_eliminate_ks)
  datap = datap[-as.numeric(as.character(rows_eliminate)),]

  bd1 = bd1[,c(datap$col)]
  order = c('patients','y','abr',names(bd1))
  order = unique(order)
  bd1 = bd1[,order]
  head(bd1[,1:10])
  dim(bd1)



  #eliminate very correlated columns
  if(!file.exists(paste(theRootDir,'correlation_pairs.txt',sep=''))){
  i_ = c()
  j_ = c()
  i1 = length(exception)+1
  i2 = dim(bd1)[2]-1

  for(i in i1:i2){
    for(j in (i+1):(dim(bd1)[2])){
      if (abs(cor(bd1[,i],bd1[,j])) >0.70){
        i_ = c(i_,i)
        j_ = c(j_,j)
      }
    }
  }

  pairs = data.frame(i=i_,j=j_)
  #write.table(pairs,paste(theRootDir,'correlation_pairs.txt',sep=''), row.names = F, sep = ';')
  }else{
    pairs = read.table(paste(theRootDir,'correlation_pairs.txt',sep=''), header = T, sep = ';')
  }


  aux0 = pairs
  keep = c()
  remove = c()


  while(dim(aux0)[1]>0 ){
    aux00 = c(aux0$i,aux0$j)
    aux1 = data.frame(table(aux00))
    aux1 = aux1[order(aux1$Freq,decreasing = TRUE),]

    keep = c(keep, as.numeric(as.character(aux1[1,1])))
    re0 = c(subset(aux0, i == as.character(aux1[1,1]))$j, subset(aux0, j == as.character(aux1[1,1]))$i)
    re0 = as.numeric(as.character(re0))
    remove = c(remove,re0)

    aux0 = subset(aux0, i!= as.character(aux1[1,1]))
    aux0 = subset(aux0, j!= as.character(aux1[1,1]))

    for(k in 1:length(re0)){
      aux0 = subset(aux0, i!=re0[k])
      aux0 = subset(aux0, j!=re0[k])
    }
  }


  datac = data.frame(col = 1:dim(bd1)[2], colname = names(bd1), rem = 0)
  datac = merge(datac, cgc, by.x='colname','Gene.Symbol',all.x=T)
  datac = datac[order(datac$col),]

  for(k in 1:length(remove)){
    if(is.na(datac[remove[k],]$Tier)){
      datac[remove[k],]$rem = 1
    }
    if(datac[remove[k],]$colname=='A1BG'){
      cat(k,remove[k])
    }
  }
  datac = subset(datac, rem==0)
  bd1 = bd1[,c(datac$col)]
  order = c('patients','y','abr',names(bd1))
  order = unique(order)
  bd1 = bd1[,order]
  head(bd1[,1:10])
  dim(bd1)

  #write.table(bd1,paste(theRootDir,'tcga_train_gexpression_cgc_7k.txt',sep=''), row.names = F, sep = ';')
}
