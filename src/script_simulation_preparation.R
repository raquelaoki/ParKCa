##########################################################
#TGP Pre-processing
##########################################################
rm(list=ls())
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

setwd("C:/Users/raoki/Documents/GitHub/ParKCa")

#reference https://github.com/knausb/vcfR
if(!require(vcfR)) install.packages('vcfR')
require(vcfR)