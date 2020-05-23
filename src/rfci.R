#trying pcalc again 
#https://cran.r-project.org/web/packages/pcalg/vignettes/pcalgDoc.pdf
#https://cran.r-project.org/web/packages/pcalg/pcalg.pdf

rm(list=ls())
library(pcalg)
require("reticulate")

setwd("~/GitHub/ParKCa/data_s")

#LOADING DATA


source_python("pickle_reader.py")
sdata <- read_pickle_file("snp_simulated1_1.txt")
 <- read_pickle_file("snp_simulated1_y01.txt")[,2]


#testing set
train = data.frame(sy01,sdata)



times = c()
ptm <- proc.time()
#SOURCE#https://cran.r-project.org/web/packages/pcalg/vignettes/pcalgDoc.pdf
#ESTIMATE CAUSAL STRUCTURE
suffStat <- list(C = cor(gmG8$x), n = nrow(gmG8$x))
pc.gmG <- pc(suffStat, indepTest = gaussCItest, p = ncol(gmG8$x), alpha = 0.01)
idaFast(causal, target, cov(gmG8$x), pc.gmG@graph)



suffStat = list(C = cor(bd), n = nrow(bd))
  model = rfci(suffStat, indepTest = gaussCItest, alpha = 0.01, m.max = 10, numCores = 3, skel.method = "stable.fast",
               conservative = FALSE, labels = names(bd))  
  aux = proc.time() - ptm
  
  
  
  
  #
  ## create the DAG :
  V <- LETTERS[1:5]
  edL <- setNames(vector("list", length = 5), V)
  edL[[1]] <- list(edges=c(2,4),weights=c(1,1))
  edL[[2]] <- list(edges=3,weights=c(1))
  edL[[3]] <- list(edges=5,weights=c(1))
  edL[[4]] <- list(edges=5,weights=c(1))
  ## and leave  edL[[ 5 ]] empty
  g <- new("graphNEL", nodes=V, edgeL=edL, edgemode="directed")
  if (require(Rgraphviz))
    plot(g)
  
  ## define the latent variable
  L <- 1
  
  ## compute the true covariance matrix of g
  cov.mat <- trueCov(g)
  ## delete rows and columns belonging to latent variable L
  true.cov <- cov.mat[-L,-L]
  ## transform covariance matrix into a correlation matrix
  true.corr <- cov2cor(true.cov)
  
  ## find PAG with RFCI algorithm
  ## as dependence "oracle", we use the true correlation matrix in
  ## gaussCItest() with a large "virtual sample size" and a large alpha :
  rfci.pag <- rfci(suffStat = list(C = true.corr, n = 10^9),
                   indepTest = gaussCItest, alpha = 0.9999, labels = V[-L],
                   verbose=TRUE)
  
  ## define PAG given in Zhang (2008), Fig. 6, p.1882
  corr.pag <- rbind(c(0,1,1,0),
                    c(1,0,0,2),
                    c(1,0,0,2),
                    c(0,3,3,0))
  ## check that estimated and correct PAG are in agreement:
  stopifnot(corr.pag == rfci.pag@amat)
  # }