# Tools for Processing and Plotting L1Nano Samples with AXOL1TL Scores using Coffea + Dask

## Setup instructions (copied from [coffea-hats repo](https://github.com/CoffeaTeam/coffea-hats))
Follow [these instructions](https://coffea-casa.readthedocs.io/en/latest/cc_user.html#access) to login to coffea-casa.
In the docker image selection dialog, select the *Coffea Base image* [**Current workspace is configured for Coffea 2025/Alma Linux 9].
Once loaded, you can clone this repository (`https://gitlab.cern.ch/cms-l1-ad/coffea-dask-axol1tl-studies.git`)
following the [git instructions](https://coffea-casa.readthedocs.io/en/latest/cc_user.html#using-git) further down in the same page.

If you choose not to use coffea-casa, be aware that any xrootd file URLs in the notebooks will need to have their file redirector changed from
`root://xcache/` to `root://cmsxrootd.fnal.gov/` or another.

Make sure you are using the *Alma 9, coffea 2025* image. 

## Running the code

`config.yaml` contains the configuration file for the dataset and options you want to run with.

To run the processor, within the coffea terminal run: 
```
python3 axo_main.py
```
This will process the files and save the output histograms for you. 

To plot the histograms you have created, take a look a `axo_plotting.ipynb`.

To save branches to parquet files, you need to first set up your grid certificate on coffea-casa. 
Open up a terminal and follow [these instructions](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookStartingGrid). 
When generating the x509 proxy, use the following commands:
`voms-proxy-init --voms cms --vomses /etc/vomses`. 
Then change the parameter `branch_writing_path` in `config.yaml` and set `save_branches` to `True`.