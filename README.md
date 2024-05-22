# Tools for Processing and Plotting L1Nano Samples with AXOL1TL Scores using Coffea + Dask

## Setup instructions (copied from [coffea-hats repo](https://github.com/CoffeaTeam/coffea-hats))
Follow [these instructions](https://coffea-casa.readthedocs.io/en/latest/cc_user.html#access) to login to coffea-casa.
In the docker image selection dialog, select the *Coffea Base image* [**Current workspace is configured for Coffea 2024/Alma Linux**].
Once loaded, you can clone this repository (`https://gitlab.cern.ch/nzipper/coffea-dask-axol1tl-studies.git`)
following the [git instructions](https://coffea-casa.readthedocs.io/en/latest/cc_user.html#using-git) further down in the same page.

If you choose not to use coffea-casa, be aware that any xrootd file URLs in the notebooks will need to have their file redirector changed from
`root://xcache/` to `root://cmsxrootd.fnal.gov/` or another.


## Using the Tools
Nothing to document really, use `axo_histos.ipynb` to plot histograms for AXOL1TL datasets!
