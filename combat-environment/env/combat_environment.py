from pettingzoo import ParallelEnv
import numpy as np
from copy import copy
from utils import calculate_potential
from gymnasium.spaces import MultiDiscrete


class CombatEnvironment(ParallelEnv):
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
        self.timestep = None
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
        - timestep
        - infos

        And any internal state used by observe() or render()
        """
        # Execute actions
        # Actions are binary vectors of the same length as the antibody/antigen sequences 
        # that specify which positions to flip
        antibody_action = actions["antibody"]
        antigen_action = actions["antigen"]

        # Update antibody sequence
        antibody_action_positions = np.where(antibody_action != 0)[0]
        self.antibody_seq[antibody_action_positions] = 1 - self.antibody_seq[antibody_action_positions]
        # Update antigen sequence
        antigen_action_positions = np.where(antigen_action != 0)[0]
        self.antigen_seq[antigen_action_positions] = 1 - self.antigen_seq[antigen_action_positions]
        
        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        potential_antibody, potential_antigen = calculate_potential(self.antibody_seq, self.antigen_seq)
        if potential_antibody == 0:
            terminations = {a: True for a in self.agents}
            rewards = {"antibody": 100, "antigen": -100}
        elif potential_antigen == 0:
            terminations = {a: True for a in self.agents}
            rewards = {"antibody": -100, "antigen": 100}
        elif potential_antibody <= potential_antigen:
            rewards = {"antibody": 1, "antigen": -1}
        else: # potential_antibody > potential_antigen
            rewards = {"antibody": -1, "antigen": 1}
        
        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            rewards = {"antibody": 0, "antigen": 0}
            truncations = {a: True for a in self.agents}

        self.timestep += 1

        # Get observations
        observations = {
            a: (self.antibody_seq, self.antigen_seq) for a in self.agents
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    def observation_space(self, agent):
        return MultiDiscrete(np.array([[2]*self.antibody_seq_len, [2]*self.antigen_seq_len]))

    def action_space(self, agent):
        """Return the action space for the agent
        Actions are binary vectors of the same length as the antibody/antigen sequences
        """
        if agent == "antibody":
            return MultiDiscrete([2]*self.antibody_seq_len)
        elif agent == "antigen":
            return MultiDiscrete([2]*self.antigen_seq_len)