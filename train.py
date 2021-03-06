import gym
import numpy as np
import torch
import torch.optim as opt
from tqdm import tqdm

import gym_puzzle
from agent import Agent


# ハイパーパラメータ
HIDDEN_NUM = 128  # エージェントの隠れ層のニューロン数
EPISODE_NUM = 10000  # エピソードを何回行うか
MAX_STEPS = 1000  # 1エピソード内で最大何回行動するか
GAMMA = .99  # 時間割引率

env = gym.make('puzzle-v0')
agent = Agent(env.metadata['N'], HIDDEN_NUM)
optimizer = opt.Adam(agent.parameters())


# 1エピソード（`done`が`True`になるまで）行動し続け、lossを返す
def do_episode():
    obs = env.reset()
    obss, actions, rewards = [], [], []

    # 現在の方策で1エピソード行動する
    agent.eval()
    with torch.no_grad():
        for step in range(1, MAX_STEPS + 1):
            # observationをPyTorchで使える形式に変換し、保存する
            obs = torch.tensor([obs], dtype=torch.float)
            obss.append(obs)

            # 方策から出力された行動確率を元に行動を決定し、行動する
            prob = agent(obs)[0].exp().numpy()
            action = np.random.choice(range(4), p=prob)
            obs, reward, done, _ = env.step(action)

            # 行動と報酬を保存する
            # 行動は、one-hot形式と呼ばれる形で保存する（後の都合）
            actions.append(torch.eye(4, dtype=torch.float)[action])
            rewards.append(reward)

            if done:
                break

    # 割引報酬和を求める
    cum_rewards = [0]
    for i, r in enumerate(rewards[::-1]):
        cum_rewards.append(GAMMA*cum_rewards[i] + r)
    cum_rewards = cum_rewards[:0:-1]

    # lossを計算して返す
    agent.train()
    loss_sum = 0
    log_pis = [agent(o)[0] * a for (o, a) in zip(obss, actions)]
    for log_pi, r in zip(log_pis, cum_rewards):
        loss_sum = loss_sum - (log_pi * r).sum()

    return loss_sum / len(obss)


if __name__ == '__main__':
    for episode in tqdm(range(1, EPISODE_NUM + 1)):
        # 1エピソードを実行して、lossを得る
        loss = do_episode()

        # lossを用いて方策を更新する
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

