#!/bin/bash

# Install requirements
pip3 install -r requirements.txt

zenodo_get 10.5281/zenodo.15405663 -o ./phenology/data/openmeteo/data

# All data should be present at this point. The script below can be run to verify whether that is the case

# Check if all data is present and download if not
# (atm automatic download is not yet implemented for GMU data and data should be obtained from the BloomBench repository)
# (downloading PEP725 data requires PEP725 credentials. see corresponding README)
# (can be commented out to skip all checks if all data is present)
#./scripts/obtain_all_data.sh
