#!/bin/bash
#SBATCH --job-name=test_ubc
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="martijncombe@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=5G
#SBATCH --time=00:20:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
 
echo "[$SHELL] #### Starting UBC test"
export CWD=$(pwd)
echo "[$SHELL] CWD: "$CWD

# Which GPU has been assigned
echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES

# Set the path to the python file
export PATH_TO_PYFILE=$CWD
echo "[$SHELL] Path of python file: "$PATH_TO_PYFILE

# Set name of the python file
export PYFILE=$CWD/exe_aqi36.py

# Run the file
echo "[$SHELL] Run script"
python exe_aqi36.py --device 'cuda:0' --num_workers 16 --modelfolder 'aqi36'
echo "[$SHELL] Script finished"

echo "[$SHELL] #### Finished UBC test. Have a nice day"