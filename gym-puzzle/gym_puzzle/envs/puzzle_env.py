import gym
import numpy as np


class PuzzleEnv(gym.Env):
    _N = 3  # パズルの1辺の長さ

    metadata = {'render.modes': ['human', 'ansi'], 'N': _N}
    observation_space = gym.spaces.Box(0, _N**2 - 1, (_N**2, ), np.uint8)
    action_space = gym.spaces.Discrete(4)
    reward_range = (-1, 1)

    _goal_field = np.arange(_N**2)  # 盤面の完成形

    def __init__(self):
        self._field = None
        self._empty_idx = None
        self._already_done = None  # 既に`done`が`True`になったか
        self._rng = None

        self.seed()

    def step(self, action):
        # 既に`done`が`True`になっているのに行動しようとした場合、
        # 盤面を動かさないでそのまま返す
        if self._already_done:
            return self._field.copy(), 0, True, {}

        # 行動できない行動を取ろうとした場合、
        # 盤面を動かさないでそのまま返す
        if self._cannot_act(action):
            return self._field.copy(), -1, True, {}

        if action == 0:  # 空きマスを上に
            self._swap(self._empty_idx, self._empty_idx - self._N)
        elif action == 1:  # 空きマスを右に
            self._swap(self._empty_idx, self._empty_idx + 1)
        elif action == 2:  # 空きマスを下に
            self._swap(self._empty_idx, self._empty_idx + self._N)
        elif action == 3:  # 空きマスを左に
            self._swap(self._empty_idx, self._empty_idx - 1)
        else:
            raise ValueError('未対応のaction')

        if self._clear():
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        return self._field.copy(), reward, done, {}

    def reset(self):
        self._already_done = False

        # fieldをランダムに初期化
        self._field = self._goal_field.copy()
        self._rng.shuffle(self._field)
        self._set_empty_idx()

        # 解なし盤面は、左右反転すると解を持つ
        if not self._solvable():
            self._fliplr()

        # 最初から完成形になっていた場合は、再度`reset`する
        return self.reset() if self._clear() else self._field.copy()

    def render(self, mode='human'):
        if mode == 'human':
            for i in range(self._N):
                print(self._field[self._N * i : self._N * (i+1)])
        elif mode == 'ansi':
            return ','.join(map(str, self._field))
        else:
            raise ValueError('未対応のmode')

    def seed(self, seed=None):
        self._rng, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # 解があるかどうかを判定する
    # 参考：<https://y-uti.hatenablog.jp/entry/2015/04/29/103422>
    def _solvable(self):
        nonzero_field = self._field[self._field.nonzero()]
        tmp = 0
        for i in range(self._N**2 - 1):
            tmp += len(np.where(nonzero_field[i + 1:] < nonzero_field[i]))
        tmp += self._empty_idx // self._N

        return tmp % 2 == 1

    # 盤面を左右反転する
    def _fliplr(self):
        for i in range(self._N):
            self._field[self._N * i : self._N * (i+1)] = np.flip(
                self._field[self._N * i : self._N * (i+1)]).copy()
        self._set_empty_idx()

    def _swap(self, idx1, idx2):
        self._field[idx1], self._field[idx2] = self._field[idx2], self._field[idx1]

    def _cannot_act(self, action):
        if action == 0:  # 空きマスを上に
            impossible_idxes = [0, 1, 2]
        elif action == 1:  # 空きマスを右に
            impossible_idxes = [2, 5, 8]
        elif action == 2:  # 空きマスを下に
            impossible_idxes = [6, 7, 8]
        else:  # 空きマスを左に
            impossible_idxes = [0, 3, 6]
        return self._empty_idx in impossible_idxes

    def _clear(self):
        return np.array_equal(self._field, self._goal_field)

    def _set_empty_idx(self):
        self._empty_idx = np.where(self._field == 0)[0][0]

