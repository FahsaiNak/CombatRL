from combat_environment import CombatEnvironment
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = CombatEnvironment()
    parallel_api_test(env, num_cycles=1_000_000)