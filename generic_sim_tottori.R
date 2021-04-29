# import required libraries
library(sp)
library(gstat)
library(stats)
library(compiler)

# import local functions
source('./nscore.R')


args = commandArgs(trailingOnly=TRUE)
if (length(args)!=6) {
	stop("Usage: Rscript generic_sim_tottori.R write_dir seed nsim dx length width", call.=FALSE)
}

# output directory
# write_dir <- '/Users/wsavran/Research/sokrg_bbp/source_models'
write_dir <- args[1]
dir.create(write_dir, showWarnings = FALSE)

# set seed
seed <- as.integer(args[2])
set.seed(seed)

# number of simulations
nsim <- as.integer(args[3])

# read data
dat <- read.csv("./rf17_mu0.csv", header=TRUE )
dat<-dat[sample(1:nrow(dat), 100000, replace=FALSE),]

# define coordinates for simulated data
coordinates(dat) = ~xx+zz
dx <- as.integer(args[4])
length <- as.integer(args[5])
width <- as.integer(args[6])
xz <- expand.grid(seq(0, length, dx), seq(0, width, dx));
names(xz) <- c("xx","zz")
gridded(xz) = ~xx+zz

# normal score transform
dat.mu0.sc <- nscore(dat$mu0)
dat$mu0.sc <- dat.mu0.sc$nscore

# load average variograms and co-variograms, saved from file analysis_two_point_average_variogram.R
load('./two_point_models/vars_rs_vario_gt_8km_co_10000_wid_250_energetic_slip_only.Rdata')

# load best fitting LMC
load('./two_point_models/lmc_250_5000_exp.Rdata')

# plot variograms to make sure everything looks good
plot(var_all, model=sim.fit)

# Create new object containing best-fitting model and conditioning data.
sim.d <- gstat( id='mu0', formula=mu0.sc~1, model=sim.fit$model$mu0, data=dat, nmax=50, maxdist=7500, beta=0 )
sim.d <- gstat( sim.d, id='slip', formula=slip.dummy~1, model=sim.fit$model$slip, dummy=TRUE, beta=0, nmax=25, maxdist=7500 )
sim.d <- gstat( sim.d, id='psv', formula=psv.dummy~1, model=sim.fit$model$psv, dummy=TRUE, beta=0, nmax=25, maxdist=7500 )
sim.d <- gstat( sim.d, id='vrup', formula=vrup.dummy~1, model=sim.fit$model$vrup, dummy=TRUE, beta=0, nmax=25, maxdist=7500 )

# enter cross-variograms
sim.d <- gstat( sim.d, id=c("slip","psv"), model=sim.fit$model$slip.psv, maxdist=7500, nmax=25 )
sim.d <- gstat( sim.d, id=c("slip","vrup"), model=sim.fit$model$slip.vrup, maxdist=7500, nmax=25 )
sim.d <- gstat( sim.d, id=c("slip","mu0"), model=sim.fit$model$slip.mu0, maxdist=7500, nmax=25 )
sim.d <- gstat( sim.d, id=c("psv","vrup"), model=sim.fit$model$psv.vrup, maxdist=7500, nmax=25 )
sim.d <- gstat( sim.d, id=c("psv","mu0"), model=sim.fit$model$psv.mu0, maxdist=7500, nmax=25 )
sim.d <- gstat( sim.d, id=c("vrup","mu0"), model=sim.fit$model$vrup.mu0, maxdist=7500, nmax=25 )

# simulate
z <- predict(sim.d, newdata=xz, nsim=nsim, debug.level = -1)

# quick and dirty plotting routine
pl1 <- spplot( z["slip.sim1"], main='Slip' )
pl2 <- spplot( z['psv.sim1'], main='Vpeak' )
pl3 <- spplot( z['vrup.sim1'], main='Vrup' )
print(pl1, split=c(1,1,2,2), more=TRUE)
print(pl2, split=c(1,2,2,2), more=TRUE)
print(pl3, split=c(2,1,2,2))
print(mean(z@data$slip.sim1))
# write binary files
for (name in colnames(z@data)) {
  out_name <- paste(gsub('\\.', '_', name), '.bin', sep="")
  print(out_name)
  file.out <- file( file.path(write_dir, out_name), 'wb')
  writeBin(z[[name]], file.out)
  flush(file.out)
  close(file.out)
}

