from .TSPEnv import TSPEnv
from .CVRPEnv import CVRPEnv
from .OPEnv import OPEnv
from .KPEnv import KPEnv
from copy import deepcopy
import numpy as np


def asign_Env(problem):
    if problem=='TSP':
        return TSPEnv
    if problem=='CVRP':
        return CVRPEnv
    if problem=='OP':
        return OPEnv
    if problem=='KP':
        return KPEnv

class COPEnv:
    def __init__(self, **env_params):
        self.env_list = []
        problem_list = list(env_params.keys())
        try:
            problem_list.remove('same')
        except:
            pass
        self.env_list = []
        for problem in problem_list:
            cop_env_list = []
            params = deepcopy(env_params[problem])
            # problem_sizes = np.argsort(env_params[problem]['problem_size'])
            for i, problem_size in enumerate(env_params[problem]['problem_size']):
                params['problem_size'] = problem_size
                params['pomo_size'] = env_params[problem]['pomo_size'][i]
                cop_env_list.append(asign_Env(problem)(**params))
            self.env_list.append(cop_env_list)


