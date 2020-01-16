import gym
import numpy as np
import tensorboardX as tbx
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
agent = Agent(env.N, HIDDEN_NUM)
optimizer = opt.Adam(agent.parameters())


def do_episode():
    obs = env.reset()
    obss, actions, rewards = [], [], []

    # 現在の方策で1エピソード行動する
    agent.eval()
    with torch.no_grad():
        for step in range(1, MAX_STEPS + 1):
            obs = torch.tensor([obs], dtype=torch.float32)
            obss.append(obs)

            prob = agent(obs)[0].exp().numpy()
            action = np.random.choice(range(4), p=prob)
            obs, reward, done, _ = env.step(action)

            actions.append(torch.eye(4, dtype=torch.float32)[action])  # 後の都合でone-hot形式で保存
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
    loss_sum = .0
    log_pis = [agent(o)[0] * a for (o, a) in zip(obss, actions)]
    for log_pi, r in zip(log_pis, cum_rewards):
        loss_sum = loss_sum - (log_pi * r).sum()

    return loss_sum / len(obss)


if __name__ == '__main__':
    with tbx.SummaryWriter() as writer:
        for episode in tqdm(range(1, EPISODE_NUM + 1)):
            loss = do_episode()

            # lossを用いて方策を更新する
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss/loss', loss.item(), episode)

    torch.save(agent.state_dict(), 'agent.tar')

