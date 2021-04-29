# import required libraries
library(sp)
library(gstat)
library(stats)
library(compiler)
# import local functions
source('./nscore.R')

# output directory
write_dir <- '~/Dropbox/Current/krg_ver1/wp_validation/krg_sources/raw_simulations/exp_rf17_full_seed1_lmc_250_5000_100m'
dir.create(write_dir, showWarnings = FALSE)
# set seed
set.seed(1)

# # read data
# dat <- read.csv("rf17_mu0.csv", header=TRUE )
# dat<-dat[sample(1:nrow(dat), 100000, replace=FALSE),]
# 
# # define coordinates for simulated data
# coordinates(dat) = ~xx+zz
# xz <- expand.grid(seq(0, 40000, 100), seq(0, 15000, 100));
# names(xz) <- c("xx","zz")
# gridded(xz) = ~xx+zz
# 
# # normal score transform
# dat.mu0.sc <- nscore(dat$mu0)
# dat$mu0.sc <- dat.mu0.sc$nscore

# load average variograms and co-variograms, saved from file analysis_two_point_average_variogram.R
load('./two_point_models/vars_rs_vario_gt_8km_co_10000_wid_250_energetic_slip_only.Rdata')

# load best fitting LMC
load('./two_point_models/lmc_250_5000_exp.Rdata')

# plot variograms to make sure everything looks good
plot(var_all, model=sim.fit)

# Create new object containing best-fitting model and conditioning data.
# sim.d <- gstat( id='mu0', formula=mu0.sc~1, model=sim.fit$model$mu0, data=dat, nmax=50, maxdist=7500, beta=0 )
# sim.d <- gstat( sim.d, id='slip', formula=slip.dummy~1, model=sim.fit$model$slip, dummy=TRUE, beta=0, nmax=25, maxdist=7500 )
# sim.d <- gstat( sim.d, id='psv', formula=psv.dummy~1, model=sim.fit$model$psv, dummy=TRUE, beta=0, nmax=25, maxdist=7500 )
# sim.d <- gstat( sim.d, id='vrup', formula=vrup.dummy~1, model=sim.fit$model$vrup, dummy=TRUE, beta=0, nmax=25, maxdist=7500 )
# 
# # enter cross-variograms
# sim.d <- gstat( sim.d, id=c("slip","psv"), model=sim.fit$model$slip.psv, maxdist=7500, nmax=25 )
# sim.d <- gstat( sim.d, id=c("slip","vrup"), model=sim.fit$model$slip.vrup, maxdist=7500, nmax=25 )
# sim.d <- gstat( sim.d, id=c("slip","mu0"), model=sim.fit$model$slip.mu0, maxdist=7500, nmax=25 )
# sim.d <- gstat( sim.d, id=c("psv","vrup"), model=sim.fit$model$psv.vrup, maxdist=7500, nmax=25 )
# sim.d <- gstat( sim.d, id=c("psv","mu0"), model=sim.fit$model$psv.mu0, maxdist=7500, nmax=25 )
# sim.d <- gstat( sim.d, id=c("vrup","mu0"), model=sim.fit$model$vrup.mu0, maxdist=7500, nmax=25 )
# 
# # simulate
# z <- predict(sim.d, newdata=xz, nsim=1, debug.level = -1)
# 
# # quick and dirty plotting routine
# pl1 <- spplot( z["slip.sim1"], main='Slip' )
# pl2 <- spplot( z['psv.sim1'], main='PSV' )
# pl3 <- spplot( z['vrup.sim1'], main='Vrup' )
# print(pl1, split=c(1,1,2,2), more=TRUE)
# print(pl2, split=c(1,2,2,2), more=TRUE)
# print(pl3, split=c(2,1,2,2))
# 
# # write binary files
# file.out <- file( file.path(write_dir, 'slip.bin'), 'wb')
# writeBin(z$slip.sim1, file.out)
# flush(file.out)
# 
# file.out <- file( file.path(write_dir, 'psv.bin'), 'wb')
# writeBin(z$psv.sim1, file.out)
# flush(file.out)
# 
# file.out <- file( file.path(write_dir, 'vrup.bin'), 'wb')
# writeBin(z$vrup.sim1, file.out)
# flush(file.out)
# 
# file.out <- file( file.path(write_dir, 'mu0.bin'), 'wb')
# writeBin(z$mu0.sim1, file.out)
# flush(file.out)

