#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#
#Plots
#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#

rm(list=ls())

require(ggplot2)
require(RColorBrewer)
require(ggrepel)
require(readxl)
require(plotROC)
require(gridExtra)
library("openxlsx")
require("reticulate")
require(Rmisc)
library(reshape2)
setwd("~/GitHub/ParKCa/results")

CREATE_CGC_BASELINES <- function(){
  #References
  #data downloaded from https://www.pnas.org/content/113/50/14330
  m_2020 = read.xlsx("~\\Documents\\GitHub\\ParKCa\\data\\driver_genes_baselines.xlsx",sheet = 1)
  m_tuson = read.xlsx("~\\Documents\\GitHub\\ParKCa\\data\\driver_genes_baselines.xlsx",sheet = 2)
  m_mutsigcv = read.xlsx("~\\Documents\\GitHub\\ParKCa\\data\\driver_genes_baselines.xlsx",sheet = 3)
  m_oncodriveFM = read.xlsx("~\\Documents\\GitHub\\ParKCa\\data\\driver_genes_baselines.xlsx",sheet = 4)
  m_oncodriveClust = read.xlsx("~\\Documents\\GitHub\\ParKCa\\data\\driver_genes_baselines.xlsx",sheet = 5)
  m_oncodriveFML = read.xlsx("~\\Documents\\GitHub\\ParKCa\\data\\driver_genes_baselines.xlsx",sheet = 6)
  m_ActiveDriver = read.xlsx("~\\Documents\\GitHub\\ParKCa\\data\\driver_genes_baselines.xlsx",sheet = 7)
  m_Music = read.xlsx("~\\Documents\\GitHub\\project\\ParKCa\\data\\driver_genes_baselines.xlsx",sheet = 8)

  cgc = read.table("~\\Documents\\GitHub\\ParKCa\\data\\cancer_gene_census.csv", header = T,sep=',')[,c(1,2)]

  #from baseline paper
  q = 0.1
  m_2020_q = subset(m_2020, m_2020[,11]<=q) #197
  m_tuson_q = subset(m_tuson, TUSON.combined.qvalue.TSG<=q) #194
  m_mutsigcv_q = m_mutsigcv[m_mutsigcv$q<q,] #158
  m_oncodriveFM_q = subset(m_oncodriveFM, QVALUE<q) #2600
  m_oncodriveClust_q = subset(m_oncodriveClust, QVALUE<q) #586
  m_oncodriveFML_q = subset(m_oncodriveFML,qvalue<q)#690
  m_ActiveDriver_q = subset(m_ActiveDriver, fdr<=q) #400
  m_Music_q = subset(m_Music, FDR.FCPT <=q) #1975

  #names
  names(cgc) = c('gene','name')

  m_2020_q$m2020 = 1
  m_tuson_q$tuson = 1
  m_mutsigcv_q$mutsigcv = 1
  m_oncodriveFM_q$oncodriveFM = 1
  m_oncodriveClust_q$oncodriveclust = 1
  m_oncodriveFML_q$oncodriveFML = 1
  m_ActiveDriver_q$activedriver = 1
  m_Music_q$music = 1

  names(m_tuson_q)[1]='gene'
  names(m_oncodriveFM_q)[1]='gene'
  names(m_oncodriveClust_q)[1]='gene'
  names(m_oncodriveFML_q)[9]='gene'
  names(m_Music_q)[1]='gene'

  m_2020_q = subset(m_2020_q, select = c('gene','m2020'))
  m_tuson_q = subset(m_tuson_q, select = c('gene','tuson'))
  m_mutsigcv_q = subset(m_mutsigcv_q, select = c('gene','mutsigcv'))
  m_oncodriveFM_q = subset(m_oncodriveFM_q, select = c('gene','oncodriveFM'))
  m_oncodriveClust_q = subset(m_oncodriveClust_q, select = c('gene','oncodriveclust'))
  m_oncodriveFML_q = subset(m_oncodriveFML_q, select = c('gene','oncodriveFML'))
  m_ActiveDriver_q = subset(m_ActiveDriver_q, select = c('gene','activedriver'))
  m_Music_q = subset(m_Music_q, select = c('gene','music'))


  data = merge(cgc,m_2020_q,all.x = TRUE)
  data = merge(data,m_tuson_q,all.x = TRUE)
  data = merge(data,m_mutsigcv_q,all.x = TRUE)
  data = merge(data,m_oncodriveFM_q,all.x = TRUE)
  data = merge(data,m_oncodriveClust_q,all.x = TRUE)
  data = merge(data,m_oncodriveFML_q,all.x = TRUE)
  data = merge(data,m_ActiveDriver_q,all.x = TRUE)
  data = merge(data,m_Music_q,all.x = TRUE)

  names(data)=c('gene','name','M2020+','TUSON','MutsigCV','oncodriveFM','OncodriveClust',
                'OncodriveFML','ActiveDriver','MuSiC')

  baselines = data.frame(colSums(data[,-c(1,2)],na.rm=TRUE))
  baselines = data.frame(rownames(baselines),baselines)
  row.names(baselines) = NULL
  names(baselines) = c('method','driver_genes')
  baselines$positives = c(dim(m_2020_q)[1],dim(m_tuson_q)[1],dim(m_mutsigcv_q)[1],
                          dim(m_oncodriveFM_q)[1],dim(m_oncodriveClust_q)[1],dim(m_oncodriveFML_q)[1],
                          dim(m_ActiveDriver_q)[1],dim(m_Music_q)[1])

  baselines$precision = baselines$driver_genes/baselines$positives
  baselines$recall = baselines$driver_genes/dim(cgc)[1]
  baselines$f1_ = 2*baselines$precision*baselines$recall/ (baselines$precision+baselines$recall)

  write.table(baselines,'~\\Documents\\GitHub\\ParKCa\\results\\cgc_baselines.txt',
              row.names = FALSE, sep = ';')

}

#Code to read python output in pickle file
source_python("pickle_reader.py")

#----------#----------#----------#----------#----------#----------#----------#
#     ROC plots - real-world dataset 
#----------#----------#----------#----------#----------#----------#----------#

roc_da <- read_pickle_file('roc_15_beforefilter.txt')
roc_bart <- read_pickle_file('roc_bart.txt')

#aux = roc_da[[2]][1] #column, row
i = 1 #from 1 to 10
data = data.frame(roc_da[[2]][i][[1]],roc_da[[3]][i][[1]])
names(data) = c('fpr','tpr')
data$sim = roc_da[[1]][i]
data$Learner = 'DA'
data$auc = roc_da[[4]][i]

for( i in 2:9){
  aux = data.frame(roc_da[[2]][i][[1]],roc_da[[3]][i][[1]])
  names(aux)=  c('fpr','tpr')
  aux$Learner = 'DA'
  aux$sim = roc_da[[1]][i]
  aux$auc = roc_da[[4]][i]
  data = rbind(data,aux)
}

for( i in 1:3){
  aux = data.frame(roc_bart[[2]][i][[1]],roc_bart[[3]][i][[1]])
  names(aux)=  c('fpr','tpr')
  aux$Learner = 'BART'
  aux$sim = roc_bart[[1]][i]
  aux$auc = roc_bart[[4]][i]
  data = rbind(data,aux)
}

data$sim = gsub('dappcalr_15_','DA+',data$sim)
data$sim = gsub('bart_','BART+',data$sim)

#type 
data$type = 'Learners Kept'
data$type[data$sim=='DA+LIHC'] = 'Learners removed'
data$type[data$sim=='DA+ESCA'] = 'Learners removed'
data$type[data$sim=='DA+PAAD'] = 'Learners removed'
data$type[data$sim=='DA+SARC'] = 'Learners removed'

#Figure 2a
g0_exp <- ggplot(data=data, aes(x=fpr, y=tpr, group=sim, col = type)) +
  geom_line(aes(linetype=type),size=1)+xlim(0,1)+ylim(0,1)+
  scale_color_manual(values = c('#009E73','#999999'))+
  scale_linetype_manual(values=c('solid','dotted'))+
  geom_abline(intercept = 0, slope = 1, color="#000000", linetype="solid")+
  theme_minimal()+xlab('False Positive Rate')+ylab('True Positive Rate')+
  #labs(caption = 'a. Test set (level 0 data)')+
  #guides(col = guide_legend(ncol =1),linetype =guide_legend(ncol =1))+
  theme(legend.position = c(0.75,0.2),
        legend.background= element_rect(fill="white",colour ="white"),
        legend.text = element_text(size=13),
        legend.key.size = unit(0.3,'cm'),
        text = element_text(size=13))+
  labs(color='',linetype='')

#Extra plots on Sup. Material 
extra1 = subset(data, sim =='BART+all' | sim =='BART+FEMALE'  | sim =='BART+MALE')
ex1 <- ggplot(data=extra1, aes(x=fpr, y=tpr, group=sim, col = sim)) +
  geom_line(aes(linetype=sim),size=1)+xlim(0,1)+ylim(0,1)+
  scale_color_manual(values = c('#00ebab','#00523b','#009E73'))+
  geom_abline(intercept = 0, slope = 1, color="#999999", linetype="solid")+
  theme_minimal()+xlab('False Positive Rate')+ylab('True Positive Rate')+
  theme(legend.position = c(0.75,0.2),
        legend.background= element_rect(fill="white",colour ="white"),
        legend.text = element_text(size=10),
        legend.key.size = unit(0.3,'cm'),
        text = element_text(size=10))+
  labs(color='',linetype='')
ex1

extra2 = subset(data, sim =='DA+all' | sim =='DA+FEMALE'  | sim =='DA+MALE')
ex2 <- ggplot(data=extra2, aes(x=fpr, y=tpr, group=sim, col = sim)) +
  geom_line(aes(linetype=sim),size=1)+xlim(0,1)+ylim(0,1)+
  scale_color_manual(values = c('#00ebab','#00523b','#009E73'))+
  geom_abline(intercept = 0, slope = 1, color="#999999", linetype="solid")+
  theme_minimal()+xlab('False Positive Rate')+ylab('True Positive Rate')+
  theme(legend.position = c(0.75,0.2),
        legend.background= element_rect(fill="white",colour ="white"),
        legend.text = element_text(size=10),
        legend.key.size = unit(0.3,'cm'),
        text = element_text(size=10))+
  labs(color='',linetype='')
ex2

extra3 = subset(data, sim =='DA+LGG' | sim =='DA+SKCM')
ex3 <- ggplot(data=extra3, aes(x=fpr, y=tpr, group=sim, col = sim)) +
  geom_line(aes(linetype=sim),size=1)+xlim(0,1)+ylim(0,1)+
  scale_color_manual(values = c('#00ebab','#00523b','#009E73'))+
  geom_abline(intercept = 0, slope = 1, color="#999999", linetype="solid")+
  theme_minimal()+xlab('False Positive Rate')+ylab('True Positive Rate')+
  theme(legend.position = c(0.75,0.2),
        legend.background= element_rect(fill="white",colour ="white"),
        legend.text = element_text(size=10),
        legend.key.size = unit(0.3,'cm'),
        text = element_text(size=10))+
  labs(color='',linetype='')
ex3

extra4 = subset(data, sim =='DA+LIHC' | sim =='DA+ESCA'|sim =='DA+PAAD' | sim =='DA+SARC')
ex4 <- ggplot(data=extra4, aes(x=fpr, y=tpr, group=sim, col = sim)) +
  geom_line(aes(linetype=sim),size=1)+xlim(0,1)+ylim(0,1)+
  scale_color_manual(values = c('#00ebab','#008560','#00523b','#00b886'))+
  geom_abline(intercept = 0, slope = 1, color="#999999", linetype="solid")+
  theme_minimal()+xlab('False Positive Rate')+ylab('True Positive Rate')+
  theme(legend.position = c(0.75,0.2),
        legend.background= element_rect(fill="white",colour ="white"),
        legend.text = element_text(size=10),
        legend.key.size = unit(0.3,'cm'),
        text = element_text(size=10))+
  labs(color='',linetype='')
ex4

#Comparing K's 
roc_da15 <- read_pickle_file('roc_15_beforefilter.txt')
roc_da30 <- read_pickle_file('roc_30_beforefilter.txt')
roc_da45 <- read_pickle_file('roc_45_beforefilter.txt')

i = 1 #from 1 to 10
#data15 = data.frame(roc_da15[[2]][i][[1]],roc_da15[[3]][i][[1]])
#data30 = data.frame(roc_da30[[2]][i][[1]],roc_da30[[3]][i][[1]])
#data45 = data.frame(roc_da45[[2]][i][[1]],roc_da45[[3]][i][[1]])

#names(data15) = names(data30) = names(data45)  = c('fpr','tpr')
data15 = data.frame('sim' = roc_da15[[1]][i])
data30 = data.frame('sim' = roc_da30[[1]][i])
data45 = data.frame('sim' = roc_da15[[1]][i])

data15$Learner = 'DA'
data30$Learner = 'DA'
data45$Learner = 'DA'

data15$auc = roc_da15[[4]][i]
data30$auc = roc_da30[[4]][i]
data45$auc = roc_da45[[4]][i]

for( i in 2:9){
  #aux15 = data.frame(roc_da15[[2]][i][[1]],roc_da15[[3]][i][[1]])
  #aux30 = data.frame(roc_da30[[2]][i][[1]],roc_da30[[3]][i][[1]])
  #aux45 = data.frame(roc_da45[[2]][i][[1]],roc_da45[[3]][i][[1]])

  #names(aux15)= names(aux30) = names(aux45) =c('fpr','tpr')
  aux15 = aux30 = aux45 = data.frame('Learner' = 'DA')

  aux15$sim = roc_da15[[1]][i]
  aux30$sim = roc_da30[[1]][i]
  aux45$sim = roc_da45[[1]][i]

  aux15$auc = roc_da15[[4]][i]
  aux30$auc = roc_da30[[4]][i]
  aux45$auc = roc_da45[[4]][i]

  data15 = rbind(data15,aux15)
  data30 = rbind(data30,aux30)
  data45 = rbind(data45,aux45)
}


data15$Learner= '15'
data30$Learner= '30' 
data45$Learner= '45'

data = rbind(data15, data30, data45)
data_s = summarySE(data, measurevar="auc", groupvars=c("Learner"))
  
ggplot(data_s, aes(x=Learner, y=auc)) + 
  geom_bar(position=position_dodge(),  fill = '#009E73', stat="identity") + 
  geom_errorbar(aes(ymin=auc-se, ymax=auc+se),width=.2, position=position_dodge(.9))+
  theme_minimal()+xlab('Number of the latent variables (k)')+ylab('Area Under the Curve (AUC)')


#----------#----------#----------#----------#----------#----------#----------#
#     ROC plots - simulation 
#----------#----------#----------#----------#----------#----------#----------#

roc_da <- read_pickle_file('sim_roc_simulations.txt')
roc_cevae <- read_pickle_file('roc_cevae.txt')

#aux = roc_da[[2]][1] #column, row
i = 1 #from 1 to 10
data = data.frame(roc_da[[2]][i][[1]],roc_da[[3]][i][[1]])
names(data) = c('fpr','tpr')
data$sim = paste(i,'da',sep='')
data$Learner = 'DA'

for( i in 2:10){
  aux = data.frame(roc_da[[2]][i][[1]],roc_da[[3]][i][[1]])
  names(aux)=  c('fpr','tpr')
  aux$Learner = 'DA'
  aux$sim = paste(i,'da',sep='')
  data = rbind(data,aux)
}

for( i in 1:10){
  aux = data.frame(roc_cevae[[2]][i][[1]],roc_cevae[[3]][i][[1]])
  names(aux)=  c('fpr','tpr')
  aux$Learner = 'CEVAE'
  aux$sim = paste(i,'cevae',sep='')
  data = rbind(data,aux)
}

#Figure 2b
g0_sim<- ggplot(data=data, aes(x=fpr, y=tpr, group=sim) ) +
  geom_line(color = '#009E73')+xlim(0,1)+ylim(0,1)+
  scale_linetype_manual(values=c('solid'))+
  geom_abline(intercept = 0, slope = 1, color="#000000", 
              linetype="solid")+
  theme_minimal()+xlab('False Positive Rate')+ylab('True Positive Rate')+
  theme(legend.position = c(0.85,0.3),
        legend.background= element_rect(fill="white",colour ="white"),
        legend.text = element_text(size=13),
        legend.key.size = unit(0.7,'cm'),
        text = element_text(size=13))


for(i in 1:10){
  i = 1+i
  colnames = paste(i,c('cevae','da'),sep='')
  extra = subset(data, sim == colnames[1]|sim == colnames[2])
  extra$sim[extra$sim==colnames[1]] = 'CEVAE'
  extra$sim[extra$sim==colnames[2]] = 'DA'
  ggplot(data=extra, aes(x=fpr, y=tpr, group=sim, color = sim) ) +
    xlim(0,1)+ylim(0,1)+ geom_line(aes(linetype=sim),size=1)+
    scale_color_manual(values = c('#00ebab','#00523b'))+
    geom_abline(intercept = 0, slope = 1, color="#999999", linetype="solid")+
    theme_minimal()+xlab('False Positive Rate')+ylab('True Positive Rate')+
    theme(legend.position = c(0.85,0.3),
          legend.background= element_rect(fill="white",colour ="white"),
          legend.text = element_text(size=10),
          legend.key.size = unit(0.7,'cm'),
          text = element_text(size=10))+
    labs(color=paste('Simulation',i,sep=' '),linetype=paste('Simulation',i,sep=' '))
} 


#----------#----------#----------#----------#----------#----------#----------#
#     Meta-learner evaluation - real-world dataset 
#----------#----------#----------#----------#----------#----------#----------#

baselines = read.table('cgc_baselines.txt',header = TRUE, sep=';')
experime1 = read.table('eval_metalevel1c.txt', header = TRUE, sep = ';')
experime0 = read.table('eval_metalevel0.txt', header = TRUE, sep = ';')

#Data prep
names(baselines)[6] = 'f1'
names(experime1)[2] = names(experime0)[2] = names(baselines)[1]
baselines$method[baselines$method=='OncodriveClust']='ODC'
baselines$method[baselines$method=='ActiveDriver']='AD'
baselines$method[baselines$method=='OncodriveFML']='ODFML'
baselines$method[baselines$method=='oncodriveFM']='ODFM'
baselines$method[baselines$method=='MuSiC']='M'


#Figure 3a: Precision x Recall: Learners, Meta-learners and random model
g1data = rbind(experime0[,c(2,3,4,7)],experime1[,c(2,3,4,7)])
g1data$name = c(rep('Learner',dim(experime0)[1]), rep('ParKCa',dim(experime1)[1]))
g1data$name[g1data$method=='random']='Random'

g1_exp <- ggplot(g1data,aes(x=precision ,y=recall,color=name,shape=name))+
geom_point(size=3)+theme_minimal() +
scale_y_continuous('Recall',limits=c(-0.09,1.05))+ #,limits=c(-0.09,1.05)
scale_x_continuous('Precision',limits=c(0,0.4))+
scale_colour_manual(values = c("#E69F00", "#56B4E9","#999999" )) + #00AFBB blue  '#9370db'(purple) '#B0C4DE'(grey),'#E7B800'(yello),'#3cb371'(green) #DB7093 (pink)
guides(size=FALSE,color=guide_legend(override.aes=list(linetype=0)))+
labs(color='',shape='')+ #,caption = 'b.Testing Set (level 0 data)'
theme(legend.position = c(0.85,0.8),
      legend.background= element_rect(fill="white",colour ="white"),
      legend.text = element_text(size=13),
      legend.key.size = unit(0.7,'cm'),
      text = element_text(size=13))


#Figure 4a: Precision x Recall - meta-learners and baselines
sub =   experime1[,c(2,8,9,6)]
names(sub) = names(baselines)[c(1,4,5, 6)]
g2data = rbind(baselines[,c(1,4,5, 6)],sub)
g2data$name = c(rep('baselines',dim(baselines)[1]), rep('meta-learners',dim(experime1)[1]))
g2data$name[g2data$method=='random']='random'  
g2data$F1 = round(g2data$f1,2)

g2data$method[g2data$method=='rf']= 'RF'
g2data$method[g2data$method=='lr']= 'LR'
g2data$method[g2data$method=='upu']= 'UPU'
g2data$method[g2data$method=='adapter']= 'Adapter'
g2data$method[g2data$method=='random']= 'Random'
g2data$method[g2data$method=='ensemble']= 'E'
g2data$method[g2data$method=='nn']= 'NN'

g2data$name[g2data$name=='meta-learners']='ParKCa'
g2data$name[g2data$name=='baselines']='Baselines'
g2data$name[g2data$name=='random']='Random'

g2_exp <- ggplot(g2data,aes(x=precision ,y=recall,color=name,shape=name))+
  geom_point(size=3)+theme_minimal() +
  scale_y_continuous('Recall',limits=c(-0.09,1.05))+ #,limits=c(-0.09,1.05)
  scale_x_continuous('Precision',limits=c(-0.09,0.7))+
  scale_colour_manual(values = c("#999999", "#56B4E9","#E69F00" )) +
  scale_fill_manual(values = c("#999999", "#56B4E9","#E69F00"))+
  guides(size=FALSE,fill = FALSE, color=guide_legend(override.aes=list(linetype=0)))+
  labs(color='',shape='')+ #,caption = 'c. Full Set (level 1 data)'
  geom_label_repel(aes(x=precision,y=recall,size=0.04,fill=name,label=method),
                   box.padding = unit(0.4, "lines"),
                   fontface='bold',color='white',segment.color = 'grey50')+
  theme(legend.position = c(0.85,0.8),
        legend.background= element_rect(fill="white",colour ="white"),
        legend.text = element_text(size=11),
        legend.key.size = unit(0.7,'cm'),
        text = element_text(size=11),
        legend.margin = margin(-0.5,0,0,0, unit="cm"))
g2_exp


#Figure 4b: F1-score metalearners and baselines 
g2data = g2data[order(g2data$F1,decreasing=TRUE),]
aux = as.character(g2data$F1)
aux[aux=='0.2']='0.20'
aux[aux=='0.1']='0.10'

g2data$method2 = paste(g2data$method, ' (',aux,')',sep='') 
g2data <- within(g2data,method2<-factor(method2,levels=g2data$method2)) 

g3_exp <- ggplot(g2data,aes(method2,F1,fill=name))+geom_bar(stat='identity')+
  theme_minimal()+labs(fill='')+
  theme(text = element_text(size=11), 
        legend.position = c(0.8, 0.8),
        legend.text = element_text(size=11),
        legend.background = element_rect(fill = 'white',linetype='solid',colour='white'),
        legend.margin = margin(-0.5,0,0,0, unit="cm"))+
  scale_fill_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
  scale_color_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
  xlab('')+ylab('F1-score')+coord_flip()
g3_exp
  
#----------#----------#----------#----------#----------#----------#----------#
#     Meta-learner evaluation - simulated datasets
#----------#----------#----------#----------#----------#----------#----------#

level0 = read.table('eval_sim_metalevel0_prob.txt', sep=';', header = T)
level1 = read.table('eval_sim_metalevel1_prob.txt', sep=';', header = T)
level1 = subset(level1, !is.na(precision))
level1c = read.table('eval_sim_metalevel1c_prob.txt', sep=';', header = T)
level1c = subset(level1c, !is.na(precision))
pehe = read.table('eval_sim_pehe_prob.txt', sep=';', header = T)

level0$metalearners[level0$metalearners=='cevae'] = 'CEVAE'
level0$metalearners[level0$metalearners=='coef'] = 'DA'

level1$metalearners[level1$metalearners=='adapter'] = 'Adapter'
level1$metalearners[level1$metalearners=='ensemble'] = 'Ensemble'
level1$metalearners[level1$metalearners=='lr'] = 'LR'
level1$metalearners[level1$metalearners=='random'] = 'Random'
level1$metalearners[level1$metalearners=='rf'] = 'RF'
level1$metalearners[level1$metalearners=='upu'] = 'UPU'
level1$metalearners[level1$metalearners=='nn'] = 'NN'


level1c$metalearners[level1c$metalearners=='adapter'] = 'Adapter'
level1c$metalearners[level1c$metalearners=='ensemble'] = 'Ensemble'
level1c$metalearners[level1c$metalearners=='lr'] = 'LR'
level1c$metalearners[level1c$metalearners=='random'] = 'Random'
level1c$metalearners[level1c$metalearners=='rf'] = 'RF'
level1c$metalearners[level1c$metalearners=='upu'] = 'UPU'
level1c$metalearners[level1c$metalearners=='nn'] = 'NN'

bdg0 = rbind(level0[,c(2,3,4)],level1[,c(2,3,4)])
p1a = data.frame(tapply(bdg0$precision,bdg0$metalearners, mean))
p1b = data.frame(tapply(bdg0$recall,bdg0$metalearners, mean))
p1c = data.frame(tapply(bdg0$precision,bdg0$metalearners, sd))
p1d = data.frame(tapply(bdg0$recall,bdg0$metalearners, sd))

p1a = data.frame(rownames(p1a),p1a); rownames(p1a)= NULL
p1b = data.frame(rownames(p1b),p1b); rownames(p1b)= NULL
p1c = data.frame(rownames(p1c),p1c); rownames(p1c)= NULL
p1d = data.frame(rownames(p1d),p1d); rownames(p1d)= NULL
names(p1a) = c('Method', 'Precision_mean')
names(p1b) = c('Method', 'Recall_mean')
names(p1c) = c('Method', 'Precision_sd')
names(p1d) = c('Method', 'Recall_sd')

p1 = merge(merge(merge(p1a,p1b),p1c),p1d)
p1$type = 'ParCKa'
p1$type[p1$Method=='CEVAE'|p1$Method=='DA']='Learner'
p1$type[p1$Method=='Random']='Random'

require(xtable)
xtable(p1)

#Figure 3b: F1-score  x % known causes for meta-learners and leaners
aux = rbind(level1[,c(2,7,11)], level0[,c(2,7,9)])
level1_s <- melt(aux, id.vars = c("metalearners",'prob'))
level1_s = summarySE(level1_s, measurevar="value", groupvars=c("variable","metalearners",'prob'))

level1_s$type = 'Meta-Learner'
level1_s$type[level1_s$metalearners =='CEVAE'|level1_s$metalearners =='DA']='Learner'
level1_s$type[level1_s$metalearners =='Random']='Random'

level1_s$F1 = as.character(round(level1_s$value,2))
level1_s = level1_s[order(level1_s$value,decreasing=TRUE),]
level1_s <- within(level1_s, metalearners<-factor( metalearners,levels= unique(level1_s$metalearners))) 

level1_s_testing = subset(level1_s, variable=='f1_')

g2_sim<- ggplot(level1_s_testing,aes(x = prob, y =value,color=metalearners, shape = metalearners))+
  geom_line(size=1) + geom_point(size=2.5)+ 
  theme_minimal()+labs(fill='')+
  scale_x_continuous(labels = scales::percent, breaks=seq(0.1,0.9,0.1))+
  guides(col =  guide_legend(ncol =3),shape = guide_legend(ncol = 3))+
  theme(text = element_text(size=11), 
        legend.position = c(0.6, 0.22),
        legend.text = element_text(size=10),
        legend.background = element_rect(fill = 'white',linetype='solid',colour='white'),
        legend.margin = margin(-0.5,0,-0.2,0, unit="cm"),
        axis.text.x = element_text(angle = 30))+
  scale_color_manual(values = c("#56B4E9","#56B4E9",'#56B4E9',"#56B4E9","#56B4E9",'#56B4E9',"#E69F00",'#E69F00',"#999999"))+
  scale_shape_manual(values=c(15,16,17,18,4,8,9,6,7))+
  xlab('Percentage known causes')+ylab('Average F1-score')+
  labs(color='',shape='')
g2_sim


#Figue 3c: pehe x % known causes x (all causes and known causes) for meta-learner x learners 
pehe2 <- melt(pehe[,c(2,3,4,5,7)], id.vars = c("method",'prob'))
pehe2$variable = as.character(pehe2$variable)
pehe2$variable[pehe2$variable=='pehe_noncausal'] = 'Non-causal Variables'
pehe2$variable[pehe2$variable=='pehe_causal'] = 'Causal Variables'
pehe2$variable[pehe2$variable=='pehe_overall'] = 'Overall'
pehe_s <- summarySE(pehe2, measurevar="value", groupvars=c("variable","method",'prob'))

pehe_s = subset(pehe_s, method!='Meta-learner (Full set)')
pehe_s$method[pehe_s$method=='Meta-learner (Testing set)'] = 'ParKCa'  
pehe_s = subset(pehe_s, variable != 'Non-causal Variables')

pehe_s_causal = subset(pehe_s, variable=='Causal Variables' )
g3_exp <- ggplot(pehe_s_causal,aes(x = prob, y =value,color=method, shape = method))+
  geom_line(size=1) + geom_point(size=2.5)+ 
  theme_minimal()+labs(fill='')+
  scale_x_continuous(labels = scales::percent, breaks=seq(0.1,0.9,0.1))+
  guides(col =  guide_legend(ncol = 1),shape = guide_legend(ncol = 1))+
  theme(text = element_text(size=10), 
        legend.position = c(0.09,0.3),
        legend.text = element_text(size=10),
        legend.background = element_rect(fill = 'white',linetype='solid',colour='white'),
        legend.margin = margin(-0.5,-0.5, -0.5,-0.5, unit="cm"),
        axis.text.x = element_text(angle = 30))+
  scale_color_manual(values = c("#E69F00",'#E69F00',"#56B4E9"))+
  scale_shape_manual(values=c(15,16,17))+
  xlab('Percentage known causes')+ylab('Av. PEHE - Causal Var.')+
  labs(color='',shape='')
g3_exp

pehe_s_causal = subset(pehe_s, variable!='Causal Variables' )

g4_exp <- ggplot(pehe_s_causal,aes(x = prob, y =value,color=method, shape = method))+
  geom_line(size=1) + geom_point(size=2.5)+ 
  scale_x_continuous(labels = scales::percent, breaks=seq(0.1,0.9,0.1))+
  theme_minimal()+labs(fill='')+
  guides(col =  FALSE,shape = FALSE)+
  theme(text = element_text(size=10), 
        legend.position = c(0.6, 0.2),
        legend.text = element_text(size=10),
        axis.text.x = element_text(angle = 30),
        legend.background = element_rect(fill = 'white',linetype='solid',colour='white'))+
  scale_color_manual(values = c("#E69F00",'#E69F00',"#56B4E9"))+
  scale_shape_manual(values=c(15,16,17))+
  xlab('Percentage of known causes')+ylab('Av. PEHE - All Var.')+
  labs(color='',shape='')
g4_exp


