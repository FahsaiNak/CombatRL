import numpy as np
import gymnasium as gym
from environment import CombatActionMaskedEnvironment
import torch
import argparse
from itertools import product
from pulp import *

def make_env(args):
    return CombatActionMaskedEnvironment(args)

def set_seed(env, seed):
    torch.manual_seed(seed)
    env.seed = seed
    np.random.seed(seed+1)
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--capacity', default='1e6', type=float)
    parser.add_argument('--steps', type=float, default=1e6)
    parser.add_argument('--max_episode_len', type=int, default=25)
    parser.add_argument('--start_steps', type=float, default=5e3)
    parser.add_argument('--evaluate-interval', type=float, default=1e3)
    parser.add_argument('--evaluate_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eps', default=1e-3, type=float)
    parser.add_argument('--adv-eps', default=1e-5, type=float)
    parser.add_argument('--num-agents', type=int, default=2)
    parser.add_argument('--num-adversaries', type=int, default=1)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=1e-5, type=float)
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--memo', default='', type=str)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_variants', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=5)
    
    args = parser.parse_args()

    max_iter = 1000
    alpha = 1e-5
    eps_start = 1.0
    eps_min = 0.05
    eps_decay = 0.99
    gamma = 0.99
    end_eps = 1e-3

    REWARD_THRESHOLD = -125
    REWARD_INCREMENT = 1

    env = make_env(args)
    rand_int = np.random.randint(1, 100)
    env = set_seed(env, seed=rand_int)

    one_state_size = args.num_variants ** args.seq_len
    one_state_spaces = np.array([i for i in product(range(args.num_variants), repeat=args.seq_len)])
    state_space_size = one_state_size ** args.num_agents
    state_spaces = np.array([i for i in product(range(args.num_variants), repeat=args.seq_len*args.num_agents)])
    action_space_size = np.prod([env.action_spaces[env.possible_agents[0]][a].n for a in env.action_spaces[env.possible_agents[0]]])
    action_spaces = [env.action_spaces[env.possible_agents[0]][a].n for a in env.action_spaces[env.possible_agents[0]]]
    action_keys = [a for a in env.action_spaces[env.possible_agents[0]]]
    Q = np.zeros((state_space_size, action_spaces[0], action_spaces[1]))
    total_rewards = {a: 0 for a in env.possible_agents}
    
    opponent_actions = np.zeros((one_state_size, action_spaces[0], action_spaces[1]))
    for _ in range(one_state_size):
        np.random.seed(rand_int)
        opponent_actions[_, np.random.choice(args.seq_len, int(args.seq_len/2), replace=False), np.random.choice(args.num_variants-1, replace=False)] = 1/(int((args.seq_len/2)))

    main_agent = env.possible_agents[0]
    opponent = env.possible_agents[1]

    for n in range(max_iter):
        Q_prev = np.copy(Q)
        state_n, _ = env.reset(seed=rand_int)
        term_n = {a: False for a in env.possible_agents}
        total_rewards_prev = total_rewards
        total_rewards = {a: 0 for a in env.possible_agents}
        actions = {a: None for a in env.possible_agents}

        if n == 0:
            epsilon = eps_start
        else:
            if total_rewards_prev[main_agent] >= REWARD_THRESHOLD:
                epsilon = max(eps_min, epsilon * eps_decay)
                REWARD_THRESHOLD = REWARD_THRESHOLD + REWARD_INCREMENT
                print(f'Epsilon: {epsilon}', f'Threshold: {REWARD_THRESHOLD}', f'Episode: {n}')
            #else:
                #epsilon = max(eps_min, epsilon * eps_decay)
            #epsilon = min(1/(eps_decay*max_iter*total_rewards[main_agent]), eps_start)
            #epsilon = max(eps_min, epsilon * eps_decay)
        
        while not any(list(term_n.values())):
            for agent in env.possible_agents:
                state_ind = np.where((state_spaces == np.hstack(state_n[agent])).all(axis=1))[0][0]
                state_opt_ind = np.where((one_state_spaces == state_n[opponent][1]).all(axis=1))[0][0]

                if agent == main_agent:
                    if np.random.uniform(0, 1) < epsilon:
                        actions[agent] = env.action_space(agent).sample()
                        _actions = [actions[agent][k] for k in action_keys]
                        _actions[1] = _actions[1] - args.num_variants - 1
                    else:
                        _action1, _action2 = np.unravel_index(Q[state_ind].argmax(), Q[state_ind].shape)
                        _actions = [_action1, _action2]
                        actions[agent] = {k: _action if k == 'position' else _action+args.num_variants+1 for k, _action in zip(action_keys, _actions)}
                    q = Q[state_ind, _actions[0], _actions[1]]
                else:
                    # _action1, _action2 = np.unravel_index(np.random.choice(np.arange(args.seq_len), p=opponent_actions[state_opt_ind].ravel()),
                    #                                       opponent_actions[state_opt_ind].shape
                    #                                       )
                    # _actions = [_action1, _action2]
                    # actions[agent] = {k: _action if k == 'position' else _action+args.num_variants+1 for k, _action in zip(action_keys, _actions)}
                    actions[agent] = env.action_space(agent).sample()

            next_state_n, reward_n, term_n, trunc_n, _ = env.step(actions)
            #print(actions)
            #print(reward_n)
        
            next_state_ind = np.where((state_spaces == np.hstack(next_state_n[main_agent])).all(axis=1))[0][0]
            Q[state_ind, _actions[0], _actions[1]] = q + (alpha * (reward_n[main_agent] + gamma * np.max(Q[next_state_ind]) - q))
            total_rewards[main_agent] += reward_n[main_agent]

            state_n = next_state_n

        #if np.max(np.abs(Q - Q_prev)) < end_eps:
            #print(f'Converged in {n} episodes')
            #break
    print(f'Episode: {n} {np.max(np.abs(Q - Q_prev))} {rand_int}')
    
    # Test
    total_reward_lst = {a: [] for a in env.possible_agents}
    for _ in range(1):
        state_n, _ = env.reset(seed=rand_int)
        print(f'Initial State: {state_n}')
        term_n = {a: False for a in env.possible_agents}
        total_rewards = {a: 0 for a in env.possible_agents}
        actions = {a: None for a in env.possible_agents}
        while not any(list(term_n.values())):
            for agent in env.possible_agents:
                state_ind = np.where((state_spaces == np.hstack(state_n[agent])).all(axis=1))[0][0]
                if agent == main_agent:
                    _action1, _action2 = np.unravel_index(Q[state_ind].argmax(), Q[state_ind].shape)
                    _actions = [_action1, _action2]
                    actions[agent] = {k: _action if k == 'position' else _action+args.num_variants+1 for k, _action in zip(action_keys, _actions)}
                else:
                    # _action1, _action2 = np.unravel_index(np.random.choice(np.arange(args.seq_len), p=opponent_actions[state_opt_ind].ravel()),
                    #                                         opponent_actions[state_opt_ind].shape
                    #                                         )
                    # _actions = [_action1, _action2]
                    # actions[agent] = {k: _action if k == 'position' else _action+args.num_variants+1 for k, _action in zip(action_keys, _actions)}
                    actions[agent] = env.action_space(agent).sample()

            next_state_n, reward_n, term_n, trunc_n, _ = env.step(actions)
            print(f'actions: {actions}', f'rewards: {reward_n}')
            state_n = next_state_n
            for a in env.possible_agents:
                total_rewards[a] += reward_n[a]
        for a in env.possible_agents:
            total_reward_lst[a].append(total_rewards[a])
        print(f'Final State: {state_n}', f'Total Reward: {total_rewards}')
    #print(f'Total Reward: {total_reward_lst}')

if __name__ == '__main__':
    main()