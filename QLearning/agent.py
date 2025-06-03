import random
from collections import defaultdict
import pickle

class QTableDefaultFactory:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def __call__(self):
        return [0.0] * self.n_actions

class QLearningAgent:
  def __init__(self, action_space, state_dim, alpha=0.5, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
    self.q_table = defaultdict(QTableDefaultFactory(len(action_space)))
    self.action_space = action_space
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_min = epsilon_min
    self.epsilon_decay = epsilon_decay

  def getAction(self, state):
    state_key = tuple(state)
    if random.random() < self.epsilon:
      return random.choice(self.action_space)
    else:
      q_values = self.q_table[state_key]
      return self.action_space[q_values.index(max(q_values))]

  def update(self, state, action, reward, next_state):
    state_key = tuple(state)
    next_state_key = tuple(next_state)
    action_index = self.action_space.index(action)

    max_next_q = max(self.q_table[next_state_key])
    current_q = self.q_table[state_key][action_index]

    new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
    self.q_table[state_key][action_index] = new_q

  def decay_epsilon(self):
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
    self.epsilon = max(self.epsilon, self.epsilon_min)

  def save(self, filepath):
    with open(filepath, 'wb') as f:
      pickle.dump(self.q_table, f)

  def load(self, filepath):
    with open(filepath, 'rb') as f:
      self.q_table = pickle.load(f)

