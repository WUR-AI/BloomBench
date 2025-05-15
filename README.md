

# BloomBench Dataset

BloomBench is a benchmark dataset for evaluating machine learning models for predicting the timing of phenological events of fruit trees. Multiple datasets are included, as well as corresponding features and baseline models. For a complete overview we refer to the paper (TBA).

## Data overview

Phenophase observations have been obtained from the Pan European Phenology Project (PEP725, http://pep725.eu/). These observations are contained in the repository.
Part of the data of George Mason University's competition on cherry tree flowering was used in this repository. To ensure reproducibility of the work we included a snapshot of the dataset that we used in this repository. The original data sources are listed in the corresponding `phenology/data/gmu_cherry/data/README.md`.
Data from the ERA5 climate reanalysis were used as meteorological features. The subset used for this dataset is hosted at [10.5281/zenodo.15405662](https://doi.org/10.5281/zenodo.15405662) (since Zenodo supports release under the original license). The codebase will download it (about 4GB) automatically.

## Usage

### Setup

BloomBench was developed and ran using Python version 3.11. CUDA version 11.4 was used.

- Run `scripts/setup.sh` to install python packages and obtain the required data.
- Optionally, run `scripts/obtain_all_data.sh` to verify whether all data is indeed present. Should any PEP725 be missing (which shouldn't be the case), the code will attempt to download it. For this it requires PEP725 login credentials. See `phenology/data/pep725/README.md` for details. In case any meteorological data is missing (which also should not be the case) these will be obtained automatically as well. At this time, missing data of the GMU repository cannot be obtained automatically.

### Run

After the setup is complete, reproduce all results in the benchmark by running `scripts/main.sh`.

Please note that in our codebase, upon loading a dataset it is stored in memory during the first iteration over all data. The first iteration will thus be slow and require patience (~a minute), but subsequent iterations are significantly faster.

Experiments were executed on a Lenovo p16 laptop (Intel Core i9-12950HX, 2300Mhz, 32GB RAM, RTX A5500 GPU) running Windows 11 Enterprise.

## License

For the PEP725 data use policy, please see [their website](http://www.pep725.eu/downloads/PEP725_Data_Use_Policy.pdf).

For GMU data, please see `phenology/data/gmu_cherry/data/README.md` for the applicable license.

For the ERA5 feature data, see the license provided at [10.5281/zenodo.15405662](https://doi.org/10.5281/zenodo.15405662).
