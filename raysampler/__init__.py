
from .montecarlo import MonteCarloRaysampler
from .allraysampler import AllRaySampler

raysampler = {
    'AllRaySampler': AllRaySampler,
    'MonteCarlo': MonteCarloRaysampler
}

# def make_raysampler(cfg):
#     return raysampler[cfg.raysampler](*cfg.raysampler)