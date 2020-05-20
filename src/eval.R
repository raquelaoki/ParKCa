#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#
#Plots
#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#

rm(list=ls())

RUN_F1_PRECISION_RECALL_SCORE = TRUE
RUN_CGC_comparison = FALSE
CREATE_CGC_BASELINES = FALSE


require(ggplot2)
require(RColorBrewer)
require(ggrepel)
require(readxl)
require(plotROC)
require(gridExtra)
library("openxlsx")

setwd("~/GitHub/ParKCa/results")


if(CREATE_CGC_BASELINES){
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

RUN_CGC_comparison = TRUE
if(RUN_CGC_comparison){
  baselines = read.table('cgc_baselines.txt',header = TRUE, sep=';')
  experime1 = read.table('eval_metalevel1.txt', header = TRUE, sep = ';')
  experime0 = read.table('eval_metalevel0.txt', header = TRUE, sep = ';')
  names(experime1)[2] = names(experime0)[2] = names(baselines)[1]
  baselines$method[baselines$method=='OncodriveClust']='ODC*'
  baselines$method[baselines$method=='ActiveDriver']='AD*'
  baselines$method[baselines$method=='OncodriveFML']='ODFML*'
  
  
  g1data = rbind(experime0[,c(2,3,4,7)],experime1[,c(2,3,4,7)])
  g1data$name = c(rep('level 0 learner',dim(experime0)[1]), rep('meta-learner',dim(experime1)[1]))
  g1data$name[g1data$method=='random']='random'
  
  
  g1 <- ggplot(g1data,aes(x=precision ,y=recall,color=name,shape=name))+
    geom_point(size=3)+theme_minimal() +
    scale_y_continuous('Recall',limits=c(-0.09,1.05))+ #,limits=c(-0.09,1.05)
    scale_x_continuous('Precision',limits=c(-0.09,0.7))+
    scale_colour_manual(values = c("#FC4E07", "#56B4E9","#E69F00" )) + #00AFBB blue  '#9370db'(purple) '#B0C4DE'(grey),'#E7B800'(yello),'#3cb371'(green) #DB7093 (pink)
    guides(size=FALSE,color=guide_legend(override.aes=list(linetype=0)))+
    labs(color='',shape='',caption = 'b.')+
    theme(legend.position = c(0.85,0.8),
          legend.background= element_rect(fill="white",colour ="white"),
          legend.text = element_text(size=12),
          legend.key.size = unit(0.7,'cm'),
          text = element_text(size=12))


  
  g2data = rbind(baselines[,c(1,4,5, 6)],experime1[,c(2,3,4,7)])
  g2data$name = c(rep('baselines',dim(baselines)[1]), rep('meta-learners',dim(experime1)[1]))
  g2data$name[g2data$method=='random']='random'  
  g2data$F1 = round(g2data$f1_,2)

  
  g2 <- ggplot(g2data,aes(x=precision ,y=recall,color=name,shape=name))+
    geom_point(size=3)+theme_minimal() +
    scale_y_continuous('Recall',limits=c(-0.09,1.05))+ #,limits=c(-0.09,1.05)
    scale_x_continuous('Precision',limits=c(-0.09,0.7))+
    scale_colour_manual(values = c("#999999", "#56B4E9","#E69F00" )) +
    scale_fill_manual(values = c("#999999", "#56B4E9","#E69F00"))+
    guides(size=FALSE,fill = FALSE, color=guide_legend(override.aes=list(linetype=0)))+
    labs(color='',shape='',caption = 'c.')+
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
          legend.position = c(0.85, 0.9),
          legend.text = element_text(size=12),
          legend.background = element_rect(fill = 'white',linetype='solid',colour='white'))+
    scale_fill_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
    scale_color_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
    xlab('')+ylab('F1-score')+coord_flip()+
    labs(caption = 'd.')
  grid.arrange(g1,g1,g2,g3, ncol=2)
  
}


dt$model_name[dt$model_name=='Logistic Regression']='LR'
dt1$model_name[dt1$model_name=='Logistic Regression']='LR'
dt1 = subset(dt1, !is.na(r))
dt$model_name = as.character(dt$model_name)

#just points
pr1<- ggplot(dt,aes(x=p,y=r,color=model_name,shape=model_name))+
  geom_point()+theme_minimal() +
  scale_y_continuous('Recall',limits=c(-0.09,1.05))+
  scale_x_continuous('Precision',limits = c(-0.09,1.05))+
  scale_shape_manual(values = c(23, 21, 24,8,12,22)) +
  scale_color_manual(values = c("#00AFBB", "#DB7093", "#FC4E07",'#B0C4DE','#E7B800','#3cb371'))+#'#9370db'
  guides(size=FALSE,color=guide_legend(override.aes=list(linetype=0)))+
  labs(color='',shape='',caption = 'e. Testing Set')+
  theme(legend.position = c(0.8,0.75),
        legend.background= element_rect(fill="white",colour ="white"),
        legend.text = element_text(size=9),
        legend.key.size = unit(0.5,'cm'),
        plot.caption = element_text(size=10))



baselines = read.table('~\\Documents\\GitHub\\project\\results\\cgc_baselines.txt',
                       header = TRUE, sep=';')
baselines = baselines[,-c(2,3)]
names(baselines) = c('Model','Precision','Recall','F1')
baselines$Type='Baseline'
dt2 = subset(dt1, select=c(model_name,p_,r_,f1_))
dt2$Type='New'
names(dt2)=names(baselines)
baselines = rbind(baselines,dt2)
baselines$Type[baselines$Model=='Random']='Random'

ggplot(baselines,aes(Precision,Recall,size=F1,color=Type))+
  geom_point()+guides(size=FALSE)+theme_minimal()+
  scale_y_continuous(limits=c(0,1))+
  scale_x_continuous(limits=c(0,1))+
  scale_color_manual(values = c("#9baec7", "#00AFBB", "#FC4E07"))+
  theme(legend.position = c(0.8,0.85),
        legend.background= element_rect(fill="white",colour ="white"))+
  geom_text_repel(baselines[baselines$Type=='New',],mapping=aes(x=Precision,y=Recall,size=0.2,label=Model,color=Type),
                  box.padding = unit(0.5, "lines"),
                  point.padding = unit(0.5, "lines"))

ggplot(baselines,aes(Precision,Recall,color=Type),size=2)+
  geom_point()+guides(size=FALSE,color=FALSE,fill=FALSE)+theme_minimal()+
  scale_y_continuous(limits=c(0,1))+
  scale_x_continuous(limits=c(0,1))+
  theme(legend.position = c(0.8,0.85),
        legend.background= element_rect(fill="white",colour ="white"))+
  scale_fill_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
  scale_color_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
  geom_label_repel(aes(x=Precision,y=Recall,size=0.04,fill=Type,label=Model),
                  box.padding = unit(0.4, "lines"),
                  fontface='bold',color='white',segment.color = 'grey50')


baselines$F1 = round(baselines$F1,2)
baselines$Model = as.character(baselines$Model)
baselines = baselines[order(baselines$F1,decreasing=TRUE),]
baselines <- within(baselines,Model<-factor(Model,levels=baselines$Model))

baselines$Type=as.character(baselines$Type)
baselines$Type[baselines$Type=='New'] = 'ParKCa'
ggplot(baselines,aes(Model,F1,fill=Type))+geom_bar(stat='identity')+
  geom_text(aes(label=F1), hjust=1.1, color="white", size=3.5)+
  theme_minimal()+labs(fill='')+
  theme(legend.position = c(0.8, 0.9),
        legend.background = element_rect(fill = 'white',linetype='solid',colour='white'))+
  scale_fill_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
  scale_color_manual(values = c("#999999",'#56B4E9',"#E69F00"))+
  xlab('')+ylab('F1-score')+coord_flip()
