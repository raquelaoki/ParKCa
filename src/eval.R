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

if(RUN_F1_PRECISION_RECALL_SCORE){
  setwd("~/GitHub/project/results")

  dt = read.table("experiments2.txt", header =T, sep = ';')[,-7]
  old = c('adapter','lr','OneClassSVM','random','randomforest','upu','ensemble')
  new = c('Adapter-PU', 'Logistic Regression','1-SVM','Random','RF','UPU','Ensemble')
  dt$model_name = as.character(dt$model_name)
  for(i in 1:length(old)){
    dt$model_name[dt$model_name==old[i]]=new[i]
  }
  dt = subset(dt, model_name!='1-SVM')
  table(dt$model_name)

  dt$ninnout = as.character(dt$nin)
  dt$ninnout[dt$nin=='[all]' & dt$nout=='[]']='Complete Data Only'
  dt$ninnout[dt$nin=='[FEMALE, MALE]' & dt$nout=='[]']='Gender Data Only'
  dt$ninnout[dt$nin=='[]' & dt$nout=='[all, FEMALE, MALE]']='Cancer Type Data Only'
  dt$ninnout[dt$nin=='[all, FEMALE, MALE]' & dt$nout=='[]']='Complete and Gender Data'
  dt$ninnout[dt$nin=='[all]' & dt$nout=='[FEMALE, MALE]']='Complete and Cancer Type Data'
  dt$ninnout[dt$nin=='[FEMALE, MALE]' & dt$nout=='[all]']='Gender and Cancer Type Data'
  dt$ninnout[dt$nin=='[]' & dt$nout=='[]']='Complete, Gender and Cancer Type Data'

  dt = subset(dt, !is.na(acc))
  dt$tnfpfntp = gsub('[','',as.character(dt$tnfpfntp), fixed = TRUE)
  dt$tnfpfntp = gsub(']','',as.character(dt$tnfpfntp),fixed = TRUE)
  dt$tnfpfntp_ = gsub('[','',as.character(dt$tnfpfntp_),fixed = TRUE)
  dt$tnfpfntp_ = gsub(']','',as.character(dt$tnfpfntp_),fixed = TRUE)

  aux = strsplit(dt$tnfpfntp,' ', fixed = TRUE)
  aux_ = strsplit(dt$tnfpfntp_,' ', fixed = TRUE)


  dt$p = ''; dt$p_ = ''
  dt$r = ''; dt$r_ = ''

  for(i in 1:dim(dt)[1]){
    aux0 = as.numeric(as.character(aux[[i]][ aux[[i]]!='']  ))
    aux0_ = as.numeric(as.character(aux_[[i]][ aux_[[i]]!='']  ))
    #tn, fp, fn, tp
    dt$p[i] = aux0[4]/(aux0[4]+aux0[2])
    dt$p_[i] = aux0_[4]/(aux0_[4]+aux0_[2])
    dt$r[i] = aux0[4]/(aux0[4]+aux0[3])
    dt$r_[i] = aux0_[4]/(aux0_[4]+aux0_[3])
  }

  dt$p = as.numeric(dt$p)
  dt$p_ = as.numeric(dt$p_)
  dt$r = as.numeric(dt$r)
  dt$r_ = as.numeric(dt$r_)

  #increase size, work on the colors and dots
  dt_pr = subset(dt,select = c(p,r,model_name))
  dt_pr$Data= 'Testing Set'
  dt_pr_ = subset(dt,select = c(p_,r_,model_name))
  dt_pr_$Data= 'Full Set'
  names(dt_pr_) = names(dt_pr)
  dt_pr = rbind(dt_pr_,dt_pr)
  dt = dt[order(dt$f1_,decreasing = TRUE),]

  #new[new=='Logistic Regression']='LR'
  dt1 = subset(dt,model_name==new[1])[1,]
  for(i in 2:length(new)){
    dt1 = rbind(dt1,subset(dt,model_name==new[i])[1,])
  }

  }

if(RUN_CAUSAL_ROC){
  #references
  #https://cran.r-project.org/web/packages/plotROC/vignettes/examples.html
  setwd("~/GitHub/project/results_k")
  #create a list of the files from your target directory
  file_list <- list.files(path="~/GitHub/project/results_k")

  flag = TRUE
  count = 0
  #had to specify columns to get rid of the total column
  for (i in 1:length(file_list)){
    file = file_list[i]
    if( strsplit(file,'_')[[1]][1] == 'roc'){
      if(flag){
        data = read.table(file, sep = ';', header = T)
        ksize =  strsplit(file,'_')[[1]][2]
        if(ksize == 'mf10' || ksize == 'pca10'|| ksize == 'ac10'){
          data$k = 10
          data$method = gsub('10','',ksize)
          data$id = file
        }
        if(ksize == 'mf20' || ksize == 'pca20'|| ksize == 'ac20'){
          data$k = 20
          data$method = gsub('20','',ksize)
          data$id = file
        }
        if(ksize == 'mf40' || ksize == 'pca40'|| ksize == 'ac40'){
          data$k = 40
          data$method = gsub('40','',ksize)
          data$id = file

        }
        if(ksize == 'mf60' || ksize == 'pca60'|| ksize == 'ac60'){
          data$k = 60
          data$method = gsub('60','',ksize)
          data$id = file

        }
        if(ksize == 'bart'){
          data$pred = 1-data$pred
          data$k = 30
          data$method = ksize
          data$id = file

        }
        flag = FALSE
        count = 1
      }else{
          data0 = read.table(file, sep = ';', header = T)
          ksize =  strsplit(file,'_')[[1]][2]
          if(ksize == 'mf10' || ksize == 'pca10'|| ksize == 'ac10'){
            data0$k = 10
            data0$method = gsub('10','',ksize)
            data0$id = file

          }
          if(ksize == 'mf20' || ksize == 'pca20'|| ksize == 'ac20'){
            data0$k = 20
            data0$method = gsub('20','',ksize)
            data0$id = file
          }
          if(ksize == 'mf40' || ksize == 'pca40'|| ksize == 'ac40'){
            data0$k = 40
            data0$method = gsub('40','',ksize)
            data0$id = file
          }
          if(ksize == 'mf60' || ksize == 'pca60'|| ksize == 'ac60'){
            data0$k = 60
            data0$method = gsub('60','',ksize)
            data0$id = file
          }
          if(ksize == 'bart'){
            data0$pred = 1-data0$pred
            data0$k = 30
            data0$method = ksize
            data0$id = file
          }
          data = rbind(data,data0)
          count = count + 1
      }

    }
  }
  require(wesanderson)
  data$k = as.factor(data$k)
  data0 = subset(data, method == 'bart' )
  data1 = subset(data, method ==  'ac' )
  data2 = subset(data, method ==  'mf' )
  data3 = subset(data, method == 'pca' )
  g0 <- ggplot(data0, aes(d = y01, m = pred, fill = id, color = k)) +
    geom_roc(show.legend = FALSE,n.cuts = 0) +
    style_roc(xlab='False Positive Rate',ylab='True Positive Rate')+
    labs(caption ='a. BART')+
    theme(plot.caption = element_text(size=10))+
    scale_color_brewer(palette = 'RdYlBu')

  g1 <- ggplot(data1, aes(d = y01, m = pred, fill = id, color = k)) +
    geom_roc(show.legend = FALSE,n.cuts = 0) +
    style_roc(xlab='False Positive Rate',ylab='True Positive Rate')+
    labs(caption ='c. DA+Autoencoder')+
    theme(plot.caption = element_text(size=10))+
    scale_color_brewer(palette = 'Oranges')
    #scale_color_manual(values=wes_palette(n=4, name="Zissou1"))

  g2 <- ggplot(data2, aes(d = y01, m = pred, fill = id, color = k)) +
    geom_roc(show.legend = FALSE,n.cuts = 0) +
    style_roc(xlab='False Positive Rate',ylab='True Positive Rate')+
    labs(caption = 'b. DA+Matrix Factorization')+
    theme(plot.caption = element_text(size=10))+
    scale_color_brewer(palette = 'Oranges')


  g3 <- ggplot(data3, aes(d = y01, m = pred, fill = id, color = k)) +
    geom_roc(show.legend = FALSE,n.cuts = 0) +
    style_roc(xlab='False Positive Rate',ylab='True Positive Rate')+
    labs(caption ='d. DA+PCA')+
    theme(plot.caption = element_text(size=10))+
    scale_color_brewer(palette = 'Oranges')

}

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

if(RUN_CGC_comparison){
  baselines = read.table('~\\Documents\\GitHub\\project\\results\\cgc_baselines.txt',
              header = TRUE, sep=';')
  baselines = subset(baselines, select = c(method,driver_genes,f1_))

  top$driver_genes = c(494,480,483,498,505,398,376,373)
  top$aux = top$model_name
  top$aux[top$aux=='Logistic Regression']='LR'
  top$aux[top$aux=='Unbiased PU']='UPU'
  top$method = paste(top$aux,c(1,2,3,4,5,1,2,3),sep='')

  top_comparison = subset(top,select=c(method,driver_genes,f1_))
  top_comparison$class = 'new'
  baselines$class = 'baseline'


  data = rbind(top_comparison,baselines)
  data = subset(data, method!='UPU3')
  data = subset(data, method!='UPU4')
  data = subset(data, method!='UPU5')
  data = subset(data, method!='LR3')

  ggplot(data=data,aes(x=f1_,y=driver_genes,color=class,label=method))+geom_point(size=2)+
    theme_minimal()+scale_x_continuous(name='F1-score',limits=c(0.08,0.28))+
    scale_y_continuous(name='Driver Genes Recovered',limits=c(45,550))+
    geom_text(hjust = 0, nudge_x = 0.002,nudge_y = 1,angle = 60,show.legend = FALSE)+
    scale_color_manual(values=c('blue','green'),guide=FALSE)


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

#top points
dt1 = subset(dt1, !is.na(model_name))
dt1$f1_ = round(dt1$f1_,2)
pr2<- ggplot(dt1,aes(x=p_,y=r_,color=model_name,shape=model_name))+
  geom_point()+ theme_minimal() +
  scale_y_continuous('Recall',limits=c(-0.09,1.05))+
  scale_x_continuous('Precision',limits = c(-0.09,1.05))+
  scale_size_continuous(range = c(2,5))+
  scale_shape_manual(values = c(23, 21, 24,8,12,22)) +
  scale_color_manual(values = c("#00AFBB", "#DB7093", "#FC4E07",'#B0C4DE','#E7B800','#3cb371'))+#'#9370db'
  guides(size=FALSE,shape=FALSE,color=FALSE,fill=FALSE)+
  labs(caption = 'f. Full set')+
  theme(plot.caption = element_text(size=10))+
  geom_label_repel(aes(x=p_,y=r_,label=f1_,fill=model_name),
                   box.padding = unit(0.4, "lines"),size=3,
                   fontface='bold',color='white',segment.color = 'grey50') +
  scale_fill_manual(values = c("#00AFBB", "#DB7093", "#FC4E07",'#B0C4DE','#E7B800','#3cb371'))

grid.arrange(g0,g1,pr1,g2,g3, pr2, ncol=3)


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
