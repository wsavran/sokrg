# import required libraries
library(sp)
library(gstat)
library(stats)
library(grid)

# import local functions
source('./nscore.R')

# gstat object data used to fit lmc (think harder about trends)
sim.g <- gstat(id='slip', formula=slip.sc~1, dummy=TRUE)
sim.g <- gstat(sim.g, 'psv', psv.sc~1, dummy=TRUE)
sim.g <- gstat(sim.g, 'vrup', vrup.sc~1, dummy=TRUE)
sim.g <- gstat(sim.g, 'mu0', mu0.sc~1, dummy=TRUE)
sim.g <- gstat(sim.g, model=vgm(,"Exp",250, add.to=vgm(,"Exp",5000)), fill.all=TRUE)

# load estimate of variograms and co-variograms, saved from file analysis_two_point_average_variogram.R
load('./two_point_models/vario_all_mean.Rdata')

# fit matern variogram to each model and fit the kappa parameter
sim.fit = fit.lmc(var_all, sim.g, correct.diagonal = 1.01)

# plot models
p<-plot(var_all, sim.fit)
print(p)