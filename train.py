import argparse
import gymnasium as gym
from environment import CombatActionMaskedEnvironment
import numpy as np
import pickle
import os
import json
import torch
import datetime
import time
from m3ddpg import M3DDPG


def make_env(args):
    return CombatActionMaskedEnvironment(args)

def make_action(actions, action_space, num_variants):
    agent_actions = {}
    agents = list(actions.keys())
    for i, agent in enumerate(agents):
        action_vec_lst = []
        for action_key, action_value in action_space(agent).items():
            action_vec = np.zeros(action_value.n)
            if action_key == 'step':
                action_vec[actions[agent][action_key]-num_variants-1] = 1
            else:
                action_vec[actions[agent][action_key]] = 1
            action_vec_lst.append(action_vec)
        stacked_action_vec = np.hstack(action_vec_lst)
        agent_actions[agent] = stacked_action_vec
    return agent_actions

def set_seed(seed, env):
    torch.manual_seed(seed)
    env.seed = seed
    np.random.seed(seed)
    return env

def evaluate(m3ddpg, args):
    env = make_env(args)
    env = set_seed(args.seed, env)
    total_rewards = {a: np.zeros(args.evaluate_num) for a in env.possible_agents}
    for n in range(args.evaluate_num):
        total_reward = {a: 0 for a in env.possible_agents}
        done_n = {a: False for a in env.possible_agents}
        state_n, info = env.reset()
        for _ in range(args.max_episode_len):
            action_n = m3ddpg.sample_action(state_n, greedy=True)
            #agent_actions = make_action(action_n, env.action_space, args.num_variants)
            next_state_n, reward_n, term_n, trunc_n, _ = env.step(action_n)
            if args.render:
                env.render()
            time.sleep(0.1)
            for i, a in enumerate(env.possible_agents):
                total_reward[a] += reward_n[a]
                if term_n[a] or trunc_n[a]:
                    done_n[a] = True
            state_n = next_state_n
            if any(list(done_n.values())):
                state_n, info = env.reset()
                break
        for a in env.possible_agents:
            total_rewards[a][n] = total_reward[a]
    if args.render:
        env.close()
    return {a: np.mean(total_rewards[a]) for a in env.possible_agents}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--capacity', default='1e6', type=float)
    parser.add_argument('--steps', type=float, default=1e6)
    parser.add_argument('--max_episode_len', type=int, default=20)
    parser.add_argument('--start_steps', type=float, default=5e3)
    parser.add_argument('--evaluate-interval', type=float, default=1e3)
    parser.add_argument('--evaluate_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eps', default=1e-5, type=float)
    parser.add_argument('--adv-eps', default=1e-3, type=float)
    parser.add_argument('--num-agents', type=int, default=2)
    parser.add_argument('--num-adversaries', type=int, default=1)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=5e-3, type=float)
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--memo', default='', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('---render', action='store_true')
    parser.add_argument('--num_variants', type=int, default=2)

    args = parser.parse_args()
    dt_now = datetime.datetime.now()
    args.logger_dir = f'experiments/{dt_now}'
    #os.makedirs(args.logger_dir, exist_ok=False)
    #with open('{}/hyperparameters.json'.format(args.logger_dir), 'w') as f:
    #    f.write(json.dumps(args.__dict__))

    env = make_env(args)
    env = set_seed(args.seed, env)
    m3ddpg = M3DDPG(args, env)

    total_reward_list = []
    reward_list = []

    step = 0
    episode_step = 0

    state_n, _ = env.reset()
    total_reward = {a: 0 for a in env.possible_agents}
    done_n = {a: False for a in env.possible_agents}
    actor_loss_list = []
    critic_loss_list = []

    while not any(list(done_n.values())):
        if step <= args.start_steps:
            action_n = {a: env.action_space(a).sample() for a in env.possible_agents}
        else:
            action_n = m3ddpg.sample_action(state_n)
        #agent_action = make_action(action_n, env.action_space, num_variants)
        next_state_n, reward_n, term_n, trunc_n, _ = env.step(action_n)
        print(term_n, trunc_n)
        for a in env.possible_agents:
            total_reward[a] += reward_n[a]
            if term_n[a] or trunc_n[a]:
                done_n[a] = True
        m3ddpg.add_memory_list(state_n, action_n, next_state_n, reward_n, done_n)
        episode_step += 1
        if step >= args.start_steps:
            actor_loss, critic_loss = m3ddpg.update()
            if actor_loss is not None:
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)

        state_n = next_state_n

        if step % args.evaluate_interval == 0:
            rewards = evaluate(m3ddpg, args)
            print('====================')
            print(f'step: {step}  reward: {rewards}')
            print('====================')
            reward_list.append(rewards)
            results = {
                'train_reward': total_reward_list,
                'reward_list': reward_list,
                }
            #pickle.dump(results, open(
            #    '{}/results.pkl'.format(args.logger_dir), 'wb'))

        if step > args.steps:
            break
        step += 1

        if any(list(done_n.values())):# or episode_step > args.max_episode_len:
            total_reward_list.append(total_reward)
            if any(list(done_n.values())) and len(actor_loss_list) != 0:
                actor_loss = np.mean(actor_loss_list, axis=0)
                critic_loss = np.mean(critic_loss_list, axis=0)
                print(f'step: {step}  reward: {total_reward}  actor loss: {actor_loss}  critic loss: {critic_loss}')
            total_reward = {a: 0 for a in env.possible_agents}
            episode_step = 0
            state_n, _ = env.reset()
            actor_loss_list = []
            critic_loss_list = []

    results = {
        'train_reward': total_reward_list,
        'reward_list': reward_list,
    }

    #pickle.dump(results, open(
    #                            '{}/results{}.pkl'.format('results', args.seed), 'wb'))

    #pickle.dump(results, open(
    #    '{}/results.pkl'.format(args.logger_dir), 'wb'))

if __name__ == '__main__':
    main()