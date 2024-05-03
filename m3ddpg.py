from model import Policy, Critic
from memory import ReplayMemory
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
import torch
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)

class Agent():
    def __init__(self, args, state_dim, action_dim_lst, name):
        self.args = args
        self.name = name
        self.actor = Policy(state_dim, action_dim_lst).to(args.device)
        self.critic = Critic(state_dim, sum(action_dim_lst)*self.args.num_agents).to(args.device)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.optimizer_actor = Adam(self.actor.parameters(), lr=args.actor_lr)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=args.critic_lr)

    def get_action(self, observation, greedy):
        action = self.actor(observation)
        if not greedy:
            action += torch.tensor(np.random.normal(0, 0.1),
                                   dtype=torch.float, device=self.args.device)
        return action

class M3DDPG():
    def __init__(self, args, env):
        self.args = args
        self.device = args.device
        self.batch_size = args.batch_size
        self.possible_agents = env.possible_agents
        self.obs_shape_n = [env.observation_spaces[a].shape[0] for a in self.possible_agents]
        self.action_keys = [k for k in env.action_spaces[self.possible_agents[0]]]
        self.action_space = [env.action_spaces[self.possible_agents[0]][k] for k in env.action_spaces[self.possible_agents[0]]]
        #num_adversaries = min(env.n, args.num_adversaries)
        self.agents = []
        for i in range(env.n):
            self.agents.append(Agent(args,
                                     sum(self.obs_shape_n),
                                     [s.n for s in self.action_space],
                                     f'{self.possible_agents[i]}'
                                    ))
        self.memory = ReplayMemory(args.capacity)

    def add_memory_list(self, *args):
        transitions = Transition(*args)
        self.memory.append(transitions)

    def sample_action(self, state, greedy=False):
        actions = {}
        for (i, a), agent in zip(enumerate(self.possible_agents), self.agents):
            observation_tensor = torch.tensor(np.hstack(state[a]),
                                              dtype=torch.float,
                                              device=self.args.device).view(-1, sum(self.obs_shape_n))
            _action = agent.get_action(observation_tensor, greedy)
            action = _action.squeeze(0).detach().cpu().numpy().tolist()
            action1, action2 = action[:self.action_space[0].n], action[self.action_space[0].n:]
            action_dict = {}
            for k, v in zip(self.action_keys, [action1, action2]):
                if k == 'step':
                    action_dict[k] = np.argmax(v)+self.args.num_variants+1
                else:
                    action_dict[k] = np.argmax(v)
            actions[a] = action_dict
        return actions

    def transition2batch(self, transitions):
        batch = Transition(*zip(*transitions))
        state_batch = torch.zeros((len(self.possible_agents), self.batch_size, sum(self.obs_shape_n)), device=self.args.device, dtype=torch.float)
        action_batch = torch.zeros((len(self.possible_agents), self.batch_size, sum([s.n for s in self.action_space])), device=self.args.device, dtype=torch.float)
        next_state_batch = torch.zeros((len(self.possible_agents), self.batch_size, sum(self.obs_shape_n)), device=self.args.device, dtype=torch.float)
        reward_batch = torch.zeros((len(self.possible_agents), self.batch_size), device=self.args.device, dtype=torch.float)
        not_done_batch = torch.zeros((len(self.possible_agents), self.batch_size), device=self.args.device, dtype=torch.float)
        for i in range(self.batch_size):
            for ai, a in enumerate(self.possible_agents):
                state_batch[ai][i] = torch.tensor(np.hstack(batch.state[i][a]))
                next_state_batch[ai][i] = torch.tensor(np.hstack(batch.next_state[i][a]))
                reward_batch[ai][i] = torch.tensor(batch.reward[i][a])
                not_done_batch[ai][i] = torch.tensor((not batch.done[i][a]))
                action_vec_lst = []
                for k, (action_key, action_value) in enumerate(batch.action[i][a].items()):
                    action_vec = np.zeros(self.action_space[k].n)
                    if action_key == 'step':
                        action_vec[batch.action[i][a][action_key]-self.args.num_variants-1] = 1
                    else:
                        action_vec[batch.action[i][a][action_key]] = 1
                    action_vec_lst.append(action_vec)
                action_batch[ai][i] = torch.tensor(np.hstack(action_vec_lst))
        return state_batch, action_batch, next_state_batch, not_done_batch, reward_batch

    def update(self):
        actor_losses, critic_losses = [], []
        if self.memory.size() <= self.args.batch_size:
            return None, None
        transitions  = self.memory.sample(self.args.batch_size)
        state_n_batch, action_n_batch, next_state_n_batch, not_done_n_batch, reward_n_batch = self.transition2batch(transitions)
        for i, agent in enumerate(self.agents):
            if 'antibody' in agent.name:
                eps = self.args.eps
            else:
                eps = self.args.adv_eps

            reward_batch = reward_n_batch[i]
            not_done_batch = not_done_n_batch[i]

            _next_actions = [self.agents[j].actor(next_state_n_batch[j]) for j in range(len(self.agents))]
            for _ in _next_actions:
                _.retain_grad()
            _next_action_n_batch_critic = torch.cat([_next_action if j != i else _next_action.detach() for j, _next_action in enumerate(_next_actions)],axis=1).squeeze(0)
            _critic_target_loss = self.agents[i].critic_target(next_state_n_batch[i], _next_action_n_batch_critic).mean()
            _critic_target_loss.backward()
            with torch.no_grad():
                next_action_n_batch_critic = torch.cat(
                    [_next_action + eps * _next_action.grad if j != i else _next_action for j, _next_action in enumerate(_next_actions)]
                    , axis=1).squeeze(0)

            _actions = [self.agents[j].actor(
                state_n_batch[j]) for j in range(len(self.agents))]
            for _ in _actions:
                _.retain_grad()
            _action_n_batch_actor = torch.cat([_action if j != i else _action.detach() for j, _action in enumerate(_actions)], axis=1)
            _actor_target_loss = self.agents[i].critic(
                state_n_batch[i], _action_n_batch_actor).mean()
            _actor_target_loss.backward(retain_graph=True)
            action_n_batch_actor = torch.cat(
                    [_action + eps * _action.grad if j != i else _action for j, _action in enumerate(_actions)], axis=1)

            ##critic
            action_n_batch_ = torch.cat([action_n_batch[j] for j in range(len(self.agents))], axis=1)
            currentQ = agent.critic(state_n_batch[i], action_n_batch_).flatten()
            nextQ = agent.critic_target(next_state_n_batch[i], next_action_n_batch_critic).flatten()
            targetQ = (reward_batch + self.args.gamma * not_done_batch * nextQ).detach()
            critic_loss = F.mse_loss(currentQ, targetQ)
            agent.optimizer_critic.zero_grad()
            critic_loss.backward()
            agent.optimizer_critic.step()

            ##policy
            actor_loss = - agent.critic(state_n_batch[i], action_n_batch_actor).mean()
            agent.optimizer_actor.zero_grad()
            actor_loss.backward()
            agent.optimizer_actor.step()

            soft_update(agent.critic_target, agent.critic, self.args.tau)
            soft_update(agent.actor_target, agent.actor, self.args.tau)

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        return actor_losses, critic_losses