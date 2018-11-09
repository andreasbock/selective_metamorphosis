from mcmc import run_mcmc
from lib import criss_cross, pringle, squeeze, triangle_flip

num_landmarks = 8
run_mcmc(*criss_cross(num_landmarks))
run_mcmc(*pringle(num_landmarks))
run_mcmc(*squeeze(num_landmarks))

num_landmarks = 9
run_mcmc(*triangle_flip(num_landmarks))
