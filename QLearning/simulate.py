from env import CourseRecEnv
from agent import QLearningAgent
import utils

def runSession():
  courses = utils.loadCourses("../assets/courses.json")
  env = CourseRecEnv(courses)
  agent = QLearningAgent(action_space=range(len(courses)), state_dim=4)
  agent.load("trained_agent.pkl")

  profile = {"major": "CS", "year": 3}
  state = env.reset(profile)

  print(f"\nRecommendations for {profile['major']} year {profile['year']} student:\n")

  for i in range(5):
    action = agent.getAction(state)
    course = courses[action]
    title = f"{course['subject']} {course['number']}: {course['title']}"
    print(f"Recommended: {title}")
    state, _, _, _ = env.step(action)

    hovered_titles = [f"{courses[idx]['subject']} {courses[idx]['number']}: {courses[idx]['title']}" for idx in env.current_state["hovered_courses"]]
    watchlist_titles = [f"{courses[idx]['subject']} {courses[idx]['number']}: {courses[idx]['title']}" for idx in env.current_state["watchlist"]]

    print(f"  Hovered courses so far: {hovered_titles}")
    print(f"  Watchlist so far: {watchlist_titles}\n")



if __name__ == "__main__":
  runSession()
