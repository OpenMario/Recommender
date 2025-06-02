from env import CourseRecEnv
from agent import QLearningAgent
import utils
import random
import matplotlib.pyplot as plt

def plotRewards(reward_history, window=50):
  plt.figure(figsize=(10, 5))
  plt.plot(reward_history, label='Reward')

  if len(reward_history) >= window:
    moving_avg = [
      sum(reward_history[i:i+window]) / window
      for i in range(len(reward_history) - window)
    ]
    plt.plot(range(window, len(reward_history)), moving_avg, label=f'{window}-episode moving avg', linestyle='--')

  plt.title("Average Reward per Episode")
  plt.xlabel("Episode")
  plt.ylabel("Total Reward")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()

def train(num_episodes=15000, actions=15):
  courses = utils.loadCourses("../assets/courses.json")
  env = CourseRecEnv(courses)
  agent = QLearningAgent(action_space=range(len(courses)), state_dim=4)

  student_profiles = [
    {"major": "CS", "year": 2},
    {"major": "cs", "year": 1},
    {"major": "BIO", "year": 3},
    {"major": "cs", "year": 4},
    {"major": "EE", "year": 2},
  ]

  reward_history = []

  for episode in range(num_episodes):
    profile = random.choice(student_profiles)
    state = env.reset(profile)
    total_reward = 0

    for step in range(actions):
      action = agent.getAction(state)
      next_state, reward, done, _ = env.step(action)
      agent.update(state, action, reward, next_state)
      state = next_state
      total_reward += reward

    reward_history.append(total_reward)
    agent.decay_epsilon()
    if episode % 100 == 0:
      print(f"Episode {episode} completed — ε = {agent.epsilon:.4f}")
  
  agent.save("trained_agent.pkl")
  print("Agent saved.")
  plotRewards(reward_history)

if __name__ == "__main__":
  train()
