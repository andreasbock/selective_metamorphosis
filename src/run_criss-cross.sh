#!/bin/bash
# Job name
#PBS -N run_mcmc_criss_cross
# Time required in hh:mm:ss
#PBS -l walltime=72:00:00
# Resource requirements
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=15999Mb
# Files to contain standard error and standard output
#PBS -o stdout
#PBS -e stderr
# Mail notification
#PBS -m ae
#PBS -M andreas.bock15@imperial.ac.uk
 
TEST_NUM=0
NUM_SAMPLES=5000
NUM_LANDMARKS=10
NUM_NUS=1
LOGDIR='results_criss_cross'
 
echo Working Directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
rm -f stdout* stderr*

# Start time
echo Start time is `date` > date
echo $HOME >> date

# source Python
module load anaconda3/personal
python $HOME/mcmc_landmarks/src/run_mcmc.py $TEST_NUM $NUM_SAMPLES $NUM_LANDMARKS $NUM_NUS $LOGDIR
mv $LOGDIR $HOME/mcmc_landmarks/

# end time
echo End time is `date` >> date