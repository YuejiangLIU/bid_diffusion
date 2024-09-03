#!/bin/bash

mkdir data && cd data

wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm -f pusht.zip

wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip
unzip robomimic_lowdim.zip && rm -f robomimic_lowdim.zip

wget https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip
unzip kitchen.zip && rm -f kitchen.zip

cd ..