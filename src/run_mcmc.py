from mcmc import run_mcmc
from lib import criss_cross, pringle, squeeze, triangle_flip

num_samples = 10

run_mcmc(*criss_cross(num_landmarks=16), num_samples)
run_mcmc(*squeeze(num_landmarks=8), num_samples)
run_mcmc(*pringle(num_landmarks=8), num_samples)
run_mcmc(*triangle_flip(num_landmarks=9), num_samples)
