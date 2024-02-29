# Tools for Processing and Plotting L1Nano Samples with AXOL1TL Scores using Coffea + Dask

## Setup instructions (copied from [coffea-hats repo](https://github.com/CoffeaTeam/coffea-hats))
Follow [these instructions](https://coffea-casa.readthedocs.io/en/latest/cc_user.html#access) to login to coffea-casa.
In the docker image selection dialog, select the *Coffea Base image*.
Once loaded, you can clone this repository (`https://gitlab.cern.ch/nzipper/coffea-dask-axol1tl-studies.git`)
following the [git instructions](https://coffea-casa.readthedocs.io/en/latest/cc_user.html#using-git) further down in the same page.

If you choose not to use coffea-casa, be aware that any xrootd file URLs in the notebooks will need to have their prefix changed from
`root://xcache/` to `root://cmsxrootd.fnal.gov/` or your favorite redirector.


## Using the Tools
Nothing to document yet, look at `histogramming_template.ipynb` for an example module!
