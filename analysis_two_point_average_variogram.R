# import required libraries
library(sp)
library(gstat)
library(stats)
# library(ggplot2)
library(grid)
# library(gridExtra)

# import local functions
source('~/Dropbox/Current/krg_ver1/nscore.R')

# read data
path = "~/Dropbox/Current/simulations/data_processed"
variogram_path <- "~/Dropbox/Current/simulations/data_processed/analysis/vario/vars_rs_vario_gt_8km_co_10000_wid_250_energetic_slip_only.Rdata"
names_sw <- dir(path, pattern="rs_pla_tap_trimmed.csv", full.names=TRUE)
names <- unique(names_sw)

vars = vector('list', length(names))
for (i in seq_along(names)) {
  print(paste('processing simulation', names[i]))
  sim <- read.csv(names[i], sep=",", header=TRUE)
  # ignore super-shear values
  sim <- sim[sim$vrup < 1.0,]
  # ignore areas in velocity strengthening area
  sim <- sim[sim$zz < 4000,]
  # only take energetically rupturing areas > mean(slip) and > mean(psv) guatteri et al, 2004
  sim <- sim[sim$slip > mean(sim$slip),]
  # sim <- sim[sim$psv > mean(sim$psv),]
  # convert to MPa
  sim$dtau <- -sim$dtau/1e6
  # reset coordinates so they are non-zero
  sim$zz <- sim$zz + abs(min(sim$zz))
  sim$xx <- sim$xx + abs(min(sim$xx))
  sim$mu0 <- abs(sim$mu0)
  # sample values if more than 10000
  if (nrow(sim) > 10000) {
    sim<-sim[sample(1:nrow(sim), 10000, replace=FALSE),]
  }
  # define coordinates for simulated data
  coordinates(sim) = ~xx+zz
  
  # normal score transform
  sim.slip.sc <- nscore(sim$slip)
  sim.psv.sc <- nscore(sim$psv)
  sim.mu0.sc <- nscore(sim$mu0)
  sim.vrup.sc <- nscore(sim$vrup)
  
  # add normal score values to data frame
  sim$slip.sc <- sim.slip.sc$nscore
  sim$psv.sc <- sim.psv.sc$nscore
  sim$mu0.sc <- sim.mu0.sc$nscore
  sim$vrup.sc <- sim.vrup.sc$nscore
  
  # gstat object data used to fit lmc (think harder about trends)
  sim.g <- gstat(id='slip', formula=slip.sc~1, data=sim)
  sim.g <- gstat(sim.g, 'psv', psv.sc~1, sim)
  sim.g <- gstat(sim.g, 'vrup', vrup.sc~1, sim)
  sim.g <- gstat(sim.g, 'mu0', mu0.sc~1, sim)
  # sim.g <- gstat(sim.g, model=vgm("Exp"), fill.all=TRUE)
  
  # estimate variograms and co-variograms
  var <- variogram(sim.g, cutoff=10000, width=250, cressie=TRUE)
  
  # append variogram to list object
  vars[[i]] <- var
}
# save variograms to file
save(vars, file=variogram_path)
save.image()

# hacky way to average across a list with annoying objects
var_all <- vars[[1]]
var_all$gamma <- 0.0
c <- 0
for (i in seq_along(vars)) {
    if (nrow(vars[[i]]) == 400) {
      c <- c + 1
      print(nrow(vars[[i]]))
      var_all$gamma <- var_all$gamma + vars[[i]]$gamma
    }
}
var_all$gamma <- var_all$gamma / c

# plot variograms
plot(var_all)
