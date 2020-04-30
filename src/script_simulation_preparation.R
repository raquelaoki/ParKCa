#########################################
# TGP data 
#########################################

rm(list=ls())
setwd("C:/Users/raque/Documents/GitHub/ParKCa")

#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#Reference: https://github.com/knausb/vcfR

if(!require(vcfR))install.packages('vcfR')

require(vcfR)

vcf <- read.vcfR( "data_s\\ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.vcf.gz", verbose = FALSE)
x <- vcfR2genlight(vcf)

### memory problem 

