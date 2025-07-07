import numpy as np
from collections import deque

class TDAgent:
    def __init__(self, no_states, no_actions, seed=42):
        self.np_random = np.random.default_rng(seed)
        self.no_states = no_states
        self.no_actions = no_actions

        self.gamma = 0.99
        self.learning_rate = 0.1

        self.v = np.zeros(self.no_states)
        #self.policy = np.zeros((self.no_states, self.no_actions))
        #self.policy[]
        self._v_history = []

    @property
    def v_history(self):
        return self._v_history

    @property
    def u(self):
        return self.gamma * self.v

    @property
    def delta(self):
        return self.u[:, np.newaxis] - self.v

    @property
    def rho(self):
        return np.array([self.no_states, 0])

    def reset(self, train: bool):
        self.train = train
        if self.train:
            self.trajectory = deque(maxlen=8)

    def step(self, observation, reward, terminated):
        # sample from behaviour policy
        action = self.np_random.choice(list(range(self.no_actions)), p=[1 - 1/self.no_states, 1/self.no_states])
        if self.train:
            self.trajectory.extend([observation, action, reward, action])
            if len(self.trajectory) >= 8:
                self.learn()
        return action

    def learn(self):
        state, _, _, action, next_state, _, terminated, _ = list(self.trajectory)
        td_error = self.delta[next_state, state]
        rho_t = self.rho[action]
        self.v += self.learning_rate * td_error * rho_t
        self._v_history.append(self.v.copy())