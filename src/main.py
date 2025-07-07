import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
from pathlib import Path
import bairds_counterexample_env

from linear_approximation import LAAgent
from temporal_difference import TDAgent


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


def la_generate(num_states: int=6,
                seed: int=42,
                max_steps: int=3000):
    env = TimeLimit(gym.make('bairds_counterexample-v0', num_intermediate_states=num_states - 1),
                    max_episode_steps=max_steps)

    agent = LAAgent(env.observation_space.n, env.action_space.n, seed)
    play_episode(env, agent, seed)
    full_w_history = np.stack(agent.w_history)

    out_path = Path("figures/la_agent") / f"{str(num_states).zfill(2)}_states" / f"{str(seed).zfill(2)}_seed" / "w_graphs" / f"w_s.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.grid()
    plt.xlabel("Steps")
    plt.ylabel("Weight")
    plt.title(f"Parameters over {num_states} States")
    plt.yscale('symlog')
    for i in range(num_states + 1):
        plt.plot(full_w_history[:, i], label=f"w_{i}")
    plt.legend(loc="best")
    plt.savefig(out_path)
    plt.close()

    for i in range(num_states + 1):
        out_path = Path("figures/la_agent") / f"{str(num_states).zfill(2)}_states" / f"{str(seed).zfill(2)}_seed" / "w_graphs" / f"w_{i}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.grid()
        plt.xlabel("Steps")
        plt.ylabel("Weight")
        plt.title(f"Parameter w_{i} over {num_states} States")
        plt.yscale('symlog')
        plt.plot(full_w_history[:, i])
        plt.savefig(out_path)
        plt.close()

    # Plot the values v
    v_history = agent.G @ full_w_history.transpose((1, 0))
    for i in range(num_states):
        out_path = Path("figures/la_agent") / f"{str(num_states).zfill(2)}_states" / f"{str(seed).zfill(2)}_seed" / "v_graphs" / f"v_{i}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.grid()
        plt.yscale('symlog')
        plt.xlabel("Steps")
        plt.ylabel("v")
        plt.plot(v_history[i, :])
        plt.savefig(out_path)
        plt.close()

def td_generate(num_states: int=6,
                seed: int=42,
                max_steps: int=3000):
    env = TimeLimit(gym.make('bairds_counterexample-v0', num_intermediate_states=num_states - 1),
                    max_episode_steps=max_steps)

    agent = TDAgent(env.observation_space.n, env.action_space.n, seed)
    play_episode(env, agent, seed)
    full_w_history = np.stack(agent.v_history)

    out_path = Path("figures/td_agent") / f"{str(num_states).zfill(2)}_states" / f"{str(seed).zfill(2)}_seed" / f"v_s.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.grid()
    plt.xlabel("Steps")
    plt.ylabel("Weight")
    plt.title(f"Parameters over {num_states} States")
    plt.yscale('symlog')
    for i in range(num_states):
        plt.plot(full_w_history[:, i], label=f"v_{i}")
    plt.legend(loc="best")
    plt.savefig(out_path)
    plt.close()

    for i in range(num_states):
        out_path = Path("figures/td_agent") / f"{str(num_states).zfill(2)}_states" / f"{str(seed).zfill(2)}_seed" / f"v_{i}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.grid()
        plt.xlabel("Steps")
        plt.ylabel("Weight")
        plt.title(f"Parameter v_{i} over {num_states} States")
        plt.yscale('symlog')
        plt.plot(full_w_history[:, i])
        plt.savefig(out_path)
        plt.close()



def main():
    for seed in [0, 42]:
        for num_states in range(2, 10):
            la_generate(num_states, seed)
            #td_generate(num_states, seed)


if __name__ == "__main__":
    main()
