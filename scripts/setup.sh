#!/bin/bash

# Install requirements
pip3 install pip-tools
pip-compile --output-file requirements.txt
pip3 install -r requirements.txt

# Obtain ERA5 meteorological features
zenodo_get 10.5281/zenodo.15405663 -o ./phenology/data/openmeteo/data

# Check if all data is present and download if not
./scripts/obtain_all_data.sh
