#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#
#Plots
#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#

rm(list=ls())

#RUN_F1_PRECISION_RECALL_SCORE = TRUE


require(ggplot2)
require(RColorBrewer)
require(ggrepel)
require(readxl)
require(plotROC)
require(gridExtra)
library("openxlsx")

setwd("~/GitHub/ParKCa/results")


#if(CREATE_CGC_BASELINES){
CREATE_CGC_BASELINES <- function(){
  #References
  #data downloaded from https://www.pnas.org/content/113/50/14330
  m_2020 = read.xlsx("~\\Documents\\GitHub\\project\\data\\driver_genes_baselines.xlsx",sheet = 1)
  m_tuson = read.xlsx("~\\Documents\\GitHub\\project\\data\\driver_genes_baselines.xlsx",sheet = 2)
  m_mutsigcv = read.xlsx("~\\Documents\\GitHub\\project\\data\\driver_genes_baselines.xlsx",sheet = 3)
  m_oncodriveFM = read.xlsx("~\\Documents\\GitHub\\project\\data\\driver_genes_baselines.xlsx",sheet = 4)
  m_oncodriveClust = read.xlsx("~\\Documents\\GitHub\\project\\data\\driver_genes_baselines.xlsx",sheet = 5)
  m_oncodriveFML = read.xlsx("~\\Documents\\GitHub\\project\\data\\driver_genes_baselines.xlsx",sheet = 6)
  m_ActiveDriver = read.xlsx("~\\Documents\\GitHub\\project\\data\\driver_genes_baselines.xlsx",sheet = 7)
  m_Music = read.xlsx("~\\Documents\\GitHub\\project\\data\\driver_genes_baselines.xlsx",sheet = 8)

  cgc = read.table("~\\Documents\\GitHub\\project\\data\\cancer_gene_census.csv", header = T,sep=',')[,c(1,2)]

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

  write.table(baselines,'~\\Documents\\GitHub\\project\\results\\cgc_baselines.txt',
              row.names = FALSE, sep = ';')

}

aplication_plots <-function(){
  baselines = read.table('cgc_baselines.txt',header = TRUE, sep=';')
  experime1 = read.table('eval_metalevel1c.txt', header = TRUE, sep = ';')
  experime0 = read.table('eval_metalevel0.txt', header = TRUE, sep = ';')
  names(baselines)[6] = 'f1'
  names(experime1)[2] = names(experime0)[2] = names(baselines)[1]
  baselines$method[baselines$method=='OncodriveClust']='ODC*'
  baselines$method[baselines$method=='ActiveDriver']='AD*'
  baselines$method[baselines$method=='OncodriveFML']='ODFML*'
  baselines$method[baselines$method=='oncodriveFM']='ODFM*'
  
  
  g1data = rbind(experime0[,c(2,3,4,7)],experime1[,c(2,3,4,7)])
  g1data$name = c(rep('Learner',dim(experime0)[1]), rep('Meta-learner',dim(experime1)[1]))
  g1data$name[g1data$method=='random']='Random'
  

  g0 <- ggplot(g1data,aes(x=precision ,y=recall,color=name,shape=name))+
    geom_point(size=3)+theme_minimal() +
    scale_y_continuous('Recall',limits=c(-0.09,1.05))+ #,limits=c(-0.09,1.05)
    scale_x_continuous('Precision',limits=c(-0.09,0.7))+
    scale_colour_manual(values = c("#FC4E07", "#56B4E9","#E69F00" )) + #00AFBB blue  '#9370db'(purple) '#B0C4DE'(grey),'#E7B800'(yello),'#3cb371'(green) #DB7093 (pink)
    guides(size=FALSE,color=guide_legend(override.aes=list(linetype=0)))+
    labs(color='',shape='',caption = 'a.Testing Set (level 0 data)')+
    theme(legend.position = c(0.85,0.8),
          legend.background= element_rect(fill="white",colour ="white"),
          legend.text = element_text(size=12),
          legend.key.size = unit(0.7,'cm'),
          text = element_text(size=12))
    
  g1 <- ggplot(g1data,aes(x=precision ,y=recall,color=name,shape=name))+
    geom_point(size=3)+theme_minimal() +
    scale_y_continuous('Recall',limits=c(-0.09,1.05))+ #,limits=c(-0.09,1.05)
    scale_x_continuous('Precision',limits=c(-0.09,0.7))+
    scale_colour_manual(values = c("#FC4E07", "#56B4E9","#E69F00" )) + #00AFBB blue  '#9370db'(purple) '#B0C4DE'(grey),'#E7B800'(yello),'#3cb371'(green) #DB7093 (pink)
    guides(size=FALSE,color=guide_legend(override.aes=list(linetype=0)))+
    labs(color='',shape='',caption = 'b.Testing Set (level 1 data)')+
    theme(legend.position = c(0.85,0.8),
          legend.background= element_rect(fill="white",colour ="white"),
          legend.text = element_text(size=12),
          legend.key.size = unit(0.7,'cm'),
          text = element_text(size=12))


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
  
  g2 <- ggplot(g2data,aes(x=precision ,y=recall,color=name,shape=name))+
    geom_point(size=3)+theme_minimal() +
    scale_y_continuous('Recall',limits=c(-0.09,1.05))+ #,limits=c(-0.09,1.05)
    scale_x_continuous('Precision',limits=c(-0.09,0.7))+
    scale_colour_manual(values = c("#999999", "#56B4E9","#E69F00" )) +
    scale_fill_manual(values = c("#999999", "#56B4E9","#E69F00"))+
    guides(size=FALSE,fill = FALSE, color=guide_legend(override.aes=list(linetype=0)))+
    labs(color='',shape='',caption = 'c. Full Set (level 1 data)')+
    geom_label_repel(aes(x=precision,y=recall,size=0.04,fill=name,label=method),
                     box.padding = unit(0.4, "lines"),
                     fontface='bold',color='white',segment.color = 'grey50')+
    theme(legend.position = c(0.85,0.8),
          legend.background= element_rect(fill="white",colour ="white"),
          legend.text = element_text(size=12),
          legend.key.size = unit(0.7,'cm'),
          text = element_text(size=12))
   
  
  g2data = g2data[order(g2data$F1,decreasing=TRUE),]
  g2data <- within(g2data,method<-factor(method,levels=g2data$method)) 
  
  g3<- ggplot(g2data,aes(method,F1,fill=name))+geom_bar(stat='identity')+
    geom_text(aes(label=F1), hjust=1.1, color="white", size=3.5)+
    theme_minimal()+labs(fill='')+
    theme(text = element_text(size=12), 
          legend.position = c(0.85, 0.75),
          legend.text = element_text(size=12),
          legend.background = element_rect(fill = 'white',linetype='solid',colour='white'))+
    scale_fill_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
    scale_color_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
    xlab('')+ylab('F1-score')+coord_flip()+
    labs(caption = 'd.Full Set (level 1 data)')
  grid.arrange(g0,g1,g2,g3, ncol=2)
  
}

require(Rmisc)
library(reshape2)

simulation_plots <-function(){
  level0 = read.table('eval_sim_metalevel0.txt', sep=';', header = T)
  level1 = read.table('eval_sim_metalevel1.txt', sep=';', header = T)
  level1c = read.table('eval_sim_metalevel1c.txt', sep=';', header = T)
  pehe = read.table('eval_sim_pehe.txt', sep=';', header = T)
  
  #NAMES
  level0$metalearners[level0$metalearners=='cevae'] = 'CEVAE'
  level0$metalearners[level0$metalearners=='coef'] = 'DA'
  
  level1$metalearners[level1$metalearners=='adapter'] = 'Adapter'
  level1$metalearners[level1$metalearners=='ensemble'] = 'Ensemble'
  level1$metalearners[level1$metalearners=='lr'] = 'LR'
  level1$metalearners[level1$metalearners=='random'] = 'Random'
  level1$metalearners[level1$metalearners=='rf'] = 'RF'
  level1$metalearners[level1$metalearners=='upu'] = 'UPU'
  
  
  level1c$metalearners[level1c$metalearners=='adapter'] = 'Adapter'
  level1c$metalearners[level1c$metalearners=='ensemble'] = 'Ensemble'
  level1c$metalearners[level1c$metalearners=='lr'] = 'LR'
  level1c$metalearners[level1c$metalearners=='random'] = 'Random'
  level1c$metalearners[level1c$metalearners=='rf'] = 'RF'
  level1c$metalearners[level1c$metalearners=='upu'] = 'UPU'
  
  #precision x score plot testing 
  #Save table for Extra Material
  #AVerage Values 
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
  p1$type = 'Meta-Learner'
  p1$type[p1$Method=='CEVAE'|p1$Method=='DA']='Learner'
  p1$type[p1$Method=='Random']='Random'
  
  #SUPPLEMENTAL MATERIAL SHOULD HAVE IT: 
  p1
  
  g0<-ggplot(p1,aes(x=Precision_mean  ,y=Recall_mean ,color=type,shape=type))+
    geom_point(size=3)+theme_minimal() +
    scale_y_continuous('Recall',limits=c(-0.09,1.05))+ #,limits=c(-0.09,1.05)
    scale_x_continuous('Precision',limits=c(-0.09,0.7))+
    scale_colour_manual(values = c("#FC4E07", "#56B4E9","#E69F00" )) + #00AFBB blue  '#9370db'(purple) '#B0C4DE'(grey),'#E7B800'(yello),'#3cb371'(green) #DB7093 (pink)
    guides(size=FALSE,color=guide_legend(override.aes=list(linetype=0)))+
    labs(color='',shape='',caption = 'a.Testing Set (level 1 data)')+
    theme(legend.position = c(0.85,0.8),
          legend.background= element_rect(fill="white",colour ="white"),
          legend.text = element_text(size=12),
          legend.key.size = unit(0.7,'cm'),
          text = element_text(size=12))
  
  
  #similar to other f1 score plot full set 
  #SAve table to other table 

  
  aux = rbind(level1[,c(2,6,7)], level0[,c(2,6,7)])
  level1_s <- melt(aux, id.vars = c("metalearners"))
  level1_s = summarySE(level1_s, measurevar="value", groupvars=c("variable","metalearners"))

  level1_s$type = 'Meta-Learner'
  level1_s$type[level1_s$metalearners =='CEVAE'|level1_s$metalearners =='DA']='Learner'
  level1_s$type[level1_s$metalearners =='Random']='Random'
  
  level1_s$F1 = as.character(round(level1_s$value,2))
  level1_s = level1_s[order(level1_s$value,decreasing=TRUE),]
  level1_s <- within(level1_s, metalearners<-factor( metalearners,levels= unique(level1_s$metalearners))) 
  
  
  level1_s_testing = subset(level1_s, variable=='f1_')
  level1_s_full = subset(level1_s, variable=='f1')
  

  g1<- ggplot(level1_s_testing,aes( metalearners,value,fill=type))+geom_bar(stat='identity')+
    geom_text(aes(label=F1,y=0.015), hjust=1.1, color="white", size=3.5)+
    geom_errorbar(aes(ymin=value-se, ymax=value+se),
                  width=.2, position=position_dodge(.9))+
    theme_minimal()+labs(fill='')+
    theme(text = element_text(size=12), 
          legend.position = c(0.85, 0.75),
          legend.text = element_text(size=12),
          legend.background = element_rect(fill = 'white',linetype='solid',colour='white'))+
    scale_fill_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
    scale_color_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
    xlab('')+ylab('F1-score')+coord_flip()+
    labs(caption = 'b.Testing Set Average (level 1 data)')
  
  g2<-ggplot(level1_s_full,aes( metalearners,value,fill=type))+geom_bar(stat='identity')+
    geom_text(aes(label=F1,y=0.02), hjust=1.1, color="white", size=3.5)+
    geom_errorbar(aes(ymin=value-se, ymax=value+se),
                  width=.2, position=position_dodge(.9))+
    theme_minimal()+labs(fill='')+
    theme(text = element_text(size=12), 
          legend.position = c(0.85, 0.75),
          legend.text = element_text(size=12),
          legend.background = element_rect(fill = 'white',linetype='solid',colour='white'))+
    scale_fill_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
    scale_color_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
    xlab('')+ylab('F1-score')+coord_flip()+
    labs(caption = 'c.Full Set Average (level 1 data)')
  
  
  pehe2 <- melt(pehe[,c(2,3,4,5)], id.vars = c("method"))
  pehe_s <- summarySE(pehe2, measurevar="value", groupvars=c("variable","method"))
  
  
  
  aux = rbind(level1c[,c(2,6,7)], level0[,c(2,6,7)])
  level1c_s <- melt(aux, id.vars = c("metalearners"))
  level1c_s = summarySE(level1c_s, measurevar="value", groupvars=c("variable","metalearners"))
  

  
  #TABLE MAYBE, different scales 
  ggplot(pehe_s, aes(x=variable, y=value, fill=method)) + 
    geom_bar(position=position_dodge(), stat="identity") +
    geom_errorbar(aes(ymin=value-se, ymax=value+se),
                  width=.2, position=position_dodge(.9))+
    theme(legend.position="top")
  
  grid.arrange(g1,g2, ncol=2)
}
