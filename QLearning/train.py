from env import CourseRecEnv
from agent import QLearningAgent
import utils

def trainAgent(num_episodes=1000, actions=10):
  # simulates 10 actions per session
  courses = utils.loadCourses("../assets/courses.json")
  env = CourseRecEnv(courses)
  agent = QLearningAgent(action_space=[1, 2, 3], state_dim=6)

  for episode in range(num_episodes):
    state = env.reset({"major": "CS", "year": 2})
    done = False

    for step in range(actions):  
      action = agent.getAction(state)
      next_state, reward, done, info = env.step(action)
      agent.update(state, action, reward, next_state)
      state = next_state

    if episode % 100 == 0:
      print(f"Episode {episode} completed")

  agent.save("trained_agent.pkl")
  print("Agent saved.")


trainAgent()