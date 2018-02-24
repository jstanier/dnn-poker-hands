#!/bin/bash

# Set up the virtualenv
virtualenv -p python3 .
source ./bin/activate

# Install dependencies
pip install tensorflow
pip install pandas
