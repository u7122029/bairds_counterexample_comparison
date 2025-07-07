from collections import deque

import numpy as np


class LAAgent:
    def __init__(self, no_states, no_actions, seed=42):
        self.np_random = np.random.default_rng(seed)

        self.no_states = no_states
        self.no_actions = no_actions

        self.gamma = 0.99
        self.learning_rate = 0.1

        self.G = np.zeros((self.no_states, self.no_states + 1))

        # Initialise G
        self.G[0] = 2
        self.G[0, -1] = 1

        id_indices = list(range(1, self.no_states))
        self.G[id_indices, id_indices] = 2
        self.G[1:, 0] = 1

        self.w = np.zeros(self.no_states + 1)
        self.w[0] = 1

        self._w_history = []

    @property
    def w_history(self):
        return self._w_history

    @property
    def v(self):
        return self.G @ self.w[:, np.newaxis]

    @property
    def u(self):
        return self.gamma * self.v

    @property
    def delta(self):
        return self.u[:, np.newaxis] - self.v

    @property
    def rho(self):
        return np.array([self.no_states, 0])

    def reset(self, train: True):
        self.train = train
        if train:
            self.trajectory = deque(maxlen=8)

    def step(self, observation, reward, terminated):
        # sampling from behaviour policy.
        action = self.np_random.choice(list(range(self.no_actions)), p=[1 - 1/self.no_states, 1/self.no_states])
        if self.train:
            self.trajectory.extend([observation, reward, terminated, action])
            if len(self.trajectory) >= 8:
                self.learn()

        return action

    def learn(self):
        state, _, _, action, next_state, _, terminated, _ = list(self.trajectory)
        # remember that reward is 0, so it doesn't matter.

        td_error = self.delta[next_state,state]
        rho_t = self.rho[action]
        self.w += self.learning_rate * td_error * rho_t * self.G[state]
        self._w_history.append(self.w.copy())

    def close(self):
        pass
