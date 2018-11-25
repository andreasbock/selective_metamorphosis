#!/bin/bash
# Job name
#PBS -N run_mcmc_squeeze
# Time required in hh:mm:ss
#PBS -l walltime=80:00:00
# Resource requirements
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=35999Mb
# Files to contain standard error and standard output
#PBS -o stdout_squeeze
#PBS -e stderr_squeeze
# Mail notification
#PBS -m ae
#PBS -M andreas.bock15@imperial.ac.uk
 
TEST_NUM=0
NUM_SAMPLES=3500
NUM_LANDMARKS=14
NUM_NUS=1
LOGDIR='results_squeeze'
 
echo Working Directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
rm -f stdout_squeeze stderr_squeeze

# Start time
echo Start time is `date` > date_squeeze
echo $HOME >> date_squeeze

# source Python
module load anaconda3/personal
python $HOME/mcmc_landmarks/src/run_mcmc.py $TEST_NUM $NUM_SAMPLES $NUM_LANDMARKS $NUM_NUS $LOGDIR
mv $LOGDIR $HOME/mcmc_landmarks/

# end time
echo End time is `date` >> date_squeeze
