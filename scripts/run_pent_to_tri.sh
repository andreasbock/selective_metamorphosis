#!/bin/bash
# Job name
#PBS -N run_mcmc_pent_to_tri
# Time required in hh:mm:ss
#PBS -l walltime=50:00:00
# Resource requirements
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=35999Mb
# Files to contain standard error and standard output
#PBS -o stdout_pent_to_tri
#PBS -e stderr_pent_to_tri
# Mail notification
#PBS -m ae
#PBS -M andreas.bock15@imperial.ac.uk
 
TEST_NUM=4
NUM_SAMPLES=2500
NUM_LANDMARKS=20
NUM_NUS=1
LOGDIR='results_pent_to_tri'
 
echo Working Directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
rm -f stdout_pent_to_tri stderr_pent_to_tri

# Start time
echo Start time is `date` > date_pent_to_tri
echo $HOME >> date_pent_to_tri

# source Python
module load anaconda3/personal
python $HOME/mcmc_landmarks/src/run_mcmc.py $TEST_NUM $NUM_SAMPLES $NUM_LANDMARKS $NUM_NUS $LOGDIR
mv $LOGDIR $HOME/mcmc_landmarks/

# end time
echo End time is `date` >> date_pent_to_tri
