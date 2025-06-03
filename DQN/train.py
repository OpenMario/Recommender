import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DQN.env import CourseRecEnv
from DQN.agent import DQNAgent
import DQN.utils as utils

import random
import matplotlib.pyplot as plt
import torch

def plotRewards(reward_history, window=50):
    """
    Plots reward history and moving average for visualization.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Reward')
    if len(reward_history) >= window:
        moving_avg = [
            sum(reward_history[i:i+window]) / window
            for i in range(len(reward_history) - window)
        ]
        plt.plot(range(window, len(reward_history)), moving_avg,
                 label=f'{window}-episode moving avg', linestyle='--')
    plt.title("Episode Reward Trend")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_dqn(num_episodes=1500, steps_per_episode=15):
    """
    Trains a Deep Q-Network agent in the course recommendation environment.
    """
    courses = utils.loadCourses("../assets/courses.json")
    env = CourseRecEnv(courses)
    state_dim = 4  # (major_encoding, year, last_hovered, last_watchlisted)
    action_size = len(courses)
    agent = DQNAgent(state_dim, action_size)

    student_profiles = [
        {"major": "CS", "year": 2},
        {"major": "ME", "year": 1},
        {"major": "BIO", "year": 3},
        {"major": "PSY", "year": 4},
        {"major": "EE", "year": 2},
    ]

    reward_history = []

    for episode in range(num_episodes):
        profile = random.choice(student_profiles)
        state = env.reset(profile)
        total_reward = 0

        for step in range(steps_per_episode):
            action = agent.getAction(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        reward_history.append(total_reward)
        agent.replay()
        agent.decay_epsilon()

        if episode % 100 == 0:
            avg = sum(reward_history[-100:]) / 100
            print(f"[Episode {episode}] AvgReward(100): {avg:.2f} | ε = {agent.epsilon:.4f}")

    
    torch.save(agent.model.state_dict(), "dqn_model.pth")
    print(" Model saved to dqn_model.pth")

    plotRewards(reward_history)

if __name__ == "__main__":
    train_dqn()
