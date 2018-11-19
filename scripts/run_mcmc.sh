#!/bin/bash
# Job name
#PBS -N my_theano_test
# Time required in hh:mm:ss
#PBS -l walltime=48:00:00
# Resource requirements
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=15999Mb
# Files to contain standard error and standard output
#PBS -o stdout
#PBS -e stderr
# Mail notification
#PBS -m ae
#PBS -M andreas.bock15@imperial.ac.uk
  
echo Working Directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
rm -f stdout* stderr*


# Start time
echo Start time is `date` > date
echo $HOME >> date

# source Python
module load anaconda3/personal

# NB: This assumes you have pulled this repository to your home directory!
python $HOME/mcmc_landmarks/src/run_mcmc.py
mv mcmc_results $HOME/mcmc_landmarks/

# end time
echo End time is `date` >> date
