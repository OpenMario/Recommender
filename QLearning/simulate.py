from env import CourseRecEnv
from agent import QLearningAgent
import utils

def runSession():
    courses = utils.loadCourses("../assets/courses.json")

    # Set up environment and agent
    env = CourseRecEnv(courses)
    agent = QLearningAgent(action_space=[1, 2, 3], state_dim=6)
    agent.load("trained_agent.pkl")

    # Simulate a session
    state = env.reset({"major": "CS", "year": 3})
    for step in range(10):
        action = agent.getAction(state)
        next_state, reward, done, info = env.step(action)
        print(f"\nStep {step+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")

        
        print(f"  Hovered Courses: {env.current_state['hovered_courses']}")
        print(f"  Watchlist: {env.current_state['watchlist']}")

        state = next_state

def getCourseTitle(course_id, courses):
    subject, number = course_id.split()
    for course in courses:
        if course["subject"] == subject and course["number"] == number:
            return course["title"]
    return "Unknown Course"


runSession()