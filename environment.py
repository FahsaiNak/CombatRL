from pettingzoo import ParallelEnv
import argparse
import functools
import numpy as np
from copy import copy
from utils import *
from gymnasium.spaces import MultiDiscrete, Dict, Discrete


class CombatActionMaskedEnvironment(ParallelEnv):
    metadata = {
        "name": "single_choice_combat",
    }

    def __init__(self, args):
        """The init method takes in environment arguments.

        Define the following attributes:
        - antibody sequence
        - antibody sequence length
        - antibody_potential
        - antigen sequence
        - antigen sequence length
        - antigen_potential
        - number of mutations per step
        - number of variants (choices for each mutation)
        - timestamp
        - possible agents
        """
        self.seed = args.seed
        self.max_episode_len = args.max_episode_len
        self.antibody_seq = None
        self.antibody_seq_len = 5
        self.antibody_potential = None
        self.antigen_seq = None
        self.antigen_seq_len = 5
        self.antigen_potential = None
        self.num_variants = args.num_variants
        self.timestep = None
        self.possible_agents = ["antibody", "antigen"]
        self.n = len(self.possible_agents)

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

        self.antibody_seq = np.zeros(self.antibody_seq_len, dtype=int) #np.random.choice([0,1], size=self.antibody_seq_len) 
        self.antigen_seq = np.zeros(self.antigen_seq_len, dtype=int) #np.random.choice([0,1], size=self.antigen_seq_len)

        #get observation with action mask for each agent
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
        antibody_prev_seq = self.antibody_seq[antibody_action["position"]]
        self.antibody_seq[antibody_action["position"]] = (antibody_prev_seq + antibody_action["step"]) % self.num_variants
        # Update antigen sequence
        antigen_prev_seq = self.antigen_seq[antigen_action["position"]]
        self.antigen_seq[antigen_action["position"]] = (antigen_prev_seq + antigen_action["step"]) % self.num_variants
        
        # Check termination conditions
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        self.antibody_potential, self.antigen_potential = calculate_potential(self.antibody_seq, self.antigen_seq)

        #best potential 
        P_best, _, _ = BestPotential(self.antibody_seq_len)

        if self.antibody_potential <= P_best:
            terminations = {'antibody': False, 'antigen': True}
            truncations = {'antibody': True, 'antigen': False}
            rewards = {"antibody": 100, "antigen": -100}
        elif self.antigen_potential <= P_best:
            terminations = {'antibody': True, 'antigen': False}
            truncations = {'antibody': False, 'antigen': True}
            rewards = {"antibody": -100, "antigen": 100}
        elif self.antibody_potential <= self.antigen_potential:
            rewards = {"antibody": 1, "antigen": -1}
        else: # self.antibody_potential > self.antigen_potential
            rewards = {"antibody": -1, "antigen": 1}
        
        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep >= self.max_episode_len:
            rewards = {"antibody": 0, "antigen": 0}
            truncations = {a: True for a in self.agents}

        self.timestep += 1

        # Get observations with action masks for each agent
        observations = {
            a: (self.antibody_seq, self.antigen_seq) for a in self.agents
        }
        
        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        confront = np.vstack((self.antibody_seq, self.antigen_seq))
        print(f"{confront} \nBinding potential: {self.antibody_potential} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent == "antibody":
            return MultiDiscrete([2]*self.antibody_seq_len)
        elif agent == "antigen":
            return MultiDiscrete([2]*self.antigen_seq_len)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Return the action space for the agent
        Actions are vectors of positions to mutate and chosen variants
        """
        return Dict({"position": Discrete(self.antibody_seq_len),
                     "step": Discrete(self.num_variants-1, start=self.num_variants+1)})

    @property
    def observation_spaces(self):
        return {a: self.observation_space(a) for a in self.possible_agents}
    
    @property
    def action_spaces(self):
        return {a: self.action_space(a) for a in self.possible_agents}