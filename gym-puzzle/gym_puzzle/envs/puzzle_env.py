import gym
import numpy as np
from gym import spaces

class PuzzleEnv(gym.Env):
    N = 3  # パズルの1辺の長さ
    metadata = {'render.modes': ['human', 'ansi']}
    observation_space = spaces.Box(0, N**2 - 1, (N**2, ), np.uint8)
    action_space = spaces.Discrete(4)
    reward_range = (-1, 1)

    goal_field = np.arange(N**2)  # 盤面の完成形

    def __init__(self):
        self.field = None
        self.empty_idx = None
        self.already_done = None
        self.rng = None

        self.seed()

    def step(self, action):
        if self.already_done:
            return self.field.copy(), 0, True, {}

        if self._cannot_act(action):
            return self.field.copy(), -1, True, {}

        if action == 0:  # 空きマスを上に
            self._swap(self.empty_idx, self.empty_idx - self.N)
        elif action == 1:  # 空きマスを右に
            self._swap(self.empty_idx, self.empty_idx + 1)
        elif action == 2:  # 空きマスを下に
            self._swap(self.empty_idx, self.empty_idx + self.N)
        elif action == 3:  # 空きマスを左に
            self._swap(self.empty_idx, self.empty_idx - 1)
        else:
            raise ValueError('未対応のaction')

        if self._clear():
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        return self.field.copy(), reward, done, {}

    def reset(self):
        # fieldをランダムに初期化（解が存在する盤面に必ずなる）
        self.field = self.goal_field.copy()
        self.rng.shuffle(self.field)
        self._set_empty_idx()
        if not self._solvable():
            self._fliplr()

        self.already_done = False
        return self.reset() if self._clear() else self.field.copy()  # 最初から完成形にならないようにしている

    def render(self, mode='human'):
        if mode == 'human':
            for i in range(self.N):
                print(self.field[self.N * i : self.N * (i+1)])
        elif mode == 'ansi':
            return ','.join(map(str, self.field))
        else:
            raise ValueError('未対応のmode')

    def seed(self, seed=None):
        self.rng, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _solvable(self):
        nonzero_field = self.field[self.field.nonzero()]
        tmp = 0
        for i in range(self.N**2 - 1):
            tmp += len(np.where(nonzero_field[i + 1:] < nonzero_field[i]))
        tmp += self.empty_idx // self.N

        return tmp % 2 == 1

    def _fliplr(self):
        # 盤面を左右反転する
        for i in range(self.N):
            self.field[self.N * i : self.N * (i+1)] = np.flip(self.field[self.N * i : self.N * (i+1)]).copy()
        self._set_empty_idx()

    def _swap(self, idx1, idx2):
        self.field[idx1], self.field[idx2] = self.field[idx2], self.field[idx1]

    def _cannot_act(self, action):
        if action == 0:  # 空きマスを上に
            impossible_idxes = [0, 1, 2]
        elif action == 1:  # 空きマスを右に
            impossible_idxes = [2, 5, 8]
        elif action == 2:  # 空きマスを下に
            impossible_idxes = [6, 7, 8]
        else:  # 空きマスを左に
            impossible_idxes = [0, 3, 6]
        return self.empty_idx in impossible_idxes

    def _clear(self):
        return np.array_equal(self.field, self.goal_field)

    def _set_empty_idx(self):
        self.empty_idx = np.where(self.field == 0)[0][0]

