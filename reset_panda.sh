#!/bin/bash

# Activate the feijoa conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate feijoa

# Run the reset commands
python panda_express/skills/actuate_gripper.py open
python panda_express/skills/go_to_conf.py
