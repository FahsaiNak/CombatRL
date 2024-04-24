from pettingzoo import ParallelEnv
import numpy as np
from copy import copy


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "combat_environment_v0",
    }

    def __init__(self):
        """The init method takes in environment arguments.

        Define the following attributes:
        - antibody sequence
        - antibody sequence length
        - antigen sequence
        - antigen sequence length
        - timestamp
        - possible agents
        """
        self.antibody_seq = None
        self.antibody_seq_len = None
        self.antigen_seq = None
        self.antigen_seq_len = None
        self.timestamp = None
        self.possible_agents = ["antibody", "antigen"]

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - antibody sequence
        - antigen sequence
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.antibody_seq = np.random.choice([0,1], size=self.antibody_seq_len)

        self.antigen_seq = np.random.choice([0,1], size=self.antigen_seq_len)

        observations = {
            a: (self.antibody_seq, self.antigen_seq) for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - antibody sequence
        - antigen sequence
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        # Execute actions
        # Actions are binary vectors of the same length as the antibody/antigen sequences 
        # that specify which positions to flip
        antibody_action = actions["antibody"]
        antigen_action = actions["antigen"]

        # Update antibody sequence
        for pos in np.where(antibody_action == 1)[0]:
            self.antibody_seq[pos] = 1 - self.antibody_seq[pos]
        # Update antigen sequence
        for pos in np.where(antigen_action == 1)[0]:
            self.antigen_seq[pos] = 1 - self.antigen_seq[pos]
        
        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]