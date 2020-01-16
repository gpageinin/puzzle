import time

import gym
import numpy as np
import torch

import gym_puzzle
from agent import Agent


# ハイパーパラメータ
HIDDEN_NUM = 128  # エージェントの隠れ層のニューロン数
MAX_STEPS = 1000  # 1エピソード内で最大何回行動するか
GAMMA = .99  # 時間割引率

env = gym.make('puzzle-v0')
agent = Agent(env.N, HIDDEN_NUM)


if __name__ == '__main__':
    agent.load_state_dict(torch.load('agent.tar'))
    obs = env.reset()

    agent.eval()
    with torch.no_grad():
        for _ in range(MAX_STEPS):
            time.sleep(1)
            env.render()
            print('-------')

            obs = torch.tensor([obs], dtype=torch.float32)

            prob = agent(obs)[0].exp().numpy()
            action = np.random.choice(range(4), p=prob)
            obs, _, done, _ = env.step(action)

            if done:
                break

