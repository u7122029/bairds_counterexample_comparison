import gymnasium as gym
import numpy as np
import bairds_counterexample_env
from gymnasium.wrappers import TimeLimit
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path


class Agent:
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
        #print(self.G.shape)
        #print(self.w[:, np.newaxis].shape)
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
        state, _, _, action, next_state, reward, terminated, _ = list(self.trajectory)

        td_error = self.delta[next_state,state]
        rho_t = self.rho[action]
        self.w += self.learning_rate * td_error * rho_t * self.G[state]
        self._w_history.append(self.w.copy())

    def close(self):
        pass


def play_episode(env, agent, seed=42):
    state, _ = env.reset(seed=seed)
    agent.reset(True)

    terminated, reward = 0, False
    episode_reward, elapsed_steps = 0, 0
    done = False

    while not done:
        action = agent.step(state, reward, terminated)
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
        done = terminated or truncated

    return episode_reward, elapsed_steps


def main():
    max_steps = 2000
    num_states = 2
    env = TimeLimit(gym.make('bairds_counterexample-v0', num_intermediate_states=num_states - 1),
                    max_episode_steps=max_steps)
    agent = Agent(env.observation_space.n, env.action_space.n, 42)
    play_episode(env, agent)
    full_w_history = np.stack(agent.w_history)

    for i in range(num_states + 1):
        out_path = Path("figures") / str(num_states).zfill(2) / f"w_{i}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.xlabel("Steps")
        plt.ylabel("Weight")
        #if num_states > 2:
        plt.yscale('symlog')
        plt.plot(full_w_history[:, i])
        plt.title(f"Parameter w_{i} over {num_states} States")
        plt.savefig(out_path)

if __name__ == "__main__":
    main()
