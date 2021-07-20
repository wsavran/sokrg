# Savran-Olsen Kinematic Rupture Generator

This repository contains the Savran-Olsen KRG along with some various utitlies and R scripts used in constructing the rupture
generator. The main KRG code can be found in `sokrg.py`. The other files are described in the Contents section below:

This rupture generator is valid for strike-slip earthquakes in the **M** 6.2 - 7.1 magnitude range. Additional work will need to
be done to generalize to other faulting types and magnitudes. 

## Installing and running KRG

Before you can run the code you need to configure the appropriate environment. We suggest using `conda` to manage the Python
environment. First, download and install [conda](https://docs.conda.io/projects/conda/en/latest/) onto your system. Next, issue
the following commands to install the required packages.
```
conda env create -f requirements.yml
conda activate sokrg
```
You will also need to download `R` and install the following packages:
- gstat

Run the rupture generator in the environment created above:

1. Clone repository
2. Navigate into repository on local computer
3. Run `sokrg.py`

Note: Modify parameters in sokrg.py for different ruptures. The ruptures are written to an .srf file or binary files containing
the parameters of the Tinti source time function.

```
git clone https://github.com/wsavran/sokrg
cd sokrg
python sokrg.py
```

## Contents

- analysis_two_point_average_variogram.R
- analysis_two_point_lmc_all.R
- central_japan_bbp1d.txt
- generic_sim.R
- generic_sim_tottori.R
- sokrg.py
- flat_earth.py
- krg_utils.py
- tinti.py
- utils.py
- 





