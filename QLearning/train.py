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

def plotRegret(regret_history, window=50):
  plt.figure(figsize=(10, 5))
  plt.plot(regret_history, label="Regret")
  if len(regret_history) >= window:
      moving_avg = [
          sum(regret_history[i:i+window]) / window
          for i in range(len(regret_history) - window)
      ]
      plt.plot(range(window, len(regret_history)), moving_avg, label=f'{window}-episode moving avg', linestyle='--')
  plt.title("Regret per Episode")
  plt.xlabel("Episode")
  plt.ylabel("Regret")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()

def plotPrecision(precision_history, window=50, k=5):
  plt.figure(figsize=(10, 5))
  plt.plot(precision_history, label=f'Precision@{k}')
  if len(precision_history) >= window:
      moving_avg = [
          sum(precision_history[i:i+window]) / window
          for i in range(len(precision_history) - window)
      ]
      plt.plot(range(window, len(precision_history)), moving_avg, label=f'{window}-episode moving avg', linestyle='--')
  plt.title(f"Precision@{k} per Episode")
  plt.xlabel("Episode")
  plt.ylabel(f"Precision@{k}")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()

def getTopActions(agent, state, k=5):
  state_key = tuple(state)
  q_values = agent.q_table.get(state_key, [0]*len(agent.action_space))  # Default zeros if unseen state
  top_k_indices = sorted(range(len(q_values)), key=lambda i: q_values[i], reverse=True)[:k]
  return top_k_indices

def precisionK(env, state, top_k_actions):
  relevant_count = 0
  for action in top_k_actions:
    course = env.courses[action]
    major_match = course.get("subject", "").lower() == env.profile["major"].lower()

    number_str = course.get("number", "100")
    try:
      number = int(number_str)
    except ValueError:
      number = 100

    year = env.profile.get("year", 1)

    if major_match and ((year == 1 and number < 300) or (year == 2 and number < 400) or (year == 3 and number < 500) or (year >= 4)):
      relevant_count += 1

  precision = relevant_count / len(top_k_actions)
  return precision

def computeRegret(env, state, action, reward):
  max_reward = 0
  for a in range(len(env.courses)):
    course = env.courses[a]
    r = 1  # base reward for hovering
    if course.get("subject", "").lower() == env.profile["major"].lower():
      r += 3
    number_str = course.get("number", "100")
    try:
      number = int(number_str)
    except ValueError:
      number = 100
    year = env.profile.get("year", 1)
    watchlist_prob = 0.3
    if course.get("subject", "").lower() == env.profile["major"].lower():
      watchlist_prob = 0.6
    if (year == 1 and number < 300) or (year == 2 and number < 400) or (year == 3 and number < 500) or (year >= 4):
      watchlist_prob += 0.2
    else:
      watchlist_prob -= 0.2
    watchlist_prob = max(0.0, min(watchlist_prob, 1.0))
    r += watchlist_prob * 5  # expected watchlist reward
    if r > max_reward:
      max_reward = r

  regret = max_reward - reward
  return regret

def train(num_episodes=3000, actions=15, precision_k=5):
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
  regret_history = []
  precision_history = []

  for episode in range(num_episodes):
    profile = random.choice(student_profiles)
    state = env.reset(profile)
    total_reward = 0
    total_regret = 0
    total_precision = 0

    for step in range(actions):
      action = agent.getAction(state)
      next_state, reward, done, _ = env.step(action)
      agent.update(state, action, reward, next_state)

      regret = computeRegret(env, state, action, reward)
      total_regret += regret

      top_k = getTopActions(agent, state, k=precision_k)
      precision = precisionK(env, state, top_k)
      total_precision += precision

      state = next_state
      total_reward += reward

    agent.decay_epsilon()

    reward_history.append(total_reward)
    regret_history.append(total_regret / actions)
    precision_history.append(total_precision / actions)

    if episode % 100 == 0:
      print(f"Episode {episode} completed — ε = {agent.epsilon:.4f}, "
        f"Reward: {total_reward:.2f}, Regret: {total_regret / actions:.3f}, "
        f"Precision@{precision_k}: {total_precision / actions:.3f}")

  agent.save("trained_agent.pkl")
  print("Agent saved.")

  plotRewards(reward_history)
  plotRegret(regret_history)
  plotPrecision(precision_history, k=precision_k)

if __name__ == "__main__":
  train()
