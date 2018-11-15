from mcmc import run_mcmc
from lib import criss_cross, pringle, squeeze, triangle_flip

num_landmarks = 8
num_samples = 500

run_mcmc(*criss_cross(num_landmarks), num_samples)
run_mcmc(*pringle(num_landmarks), num_samples)
run_mcmc(*squeeze(num_landmarks), num_samples)

num_landmarks = 9
run_mcmc(*triangle_flip(num_landmarks), num_samples)
