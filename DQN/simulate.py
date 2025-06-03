import torch
from DQN.env import CourseRecEnv
from DQN.agent import DQN, DQNAgent  
import DQN.utils as utils
import numpy as np

def runSession():

    courses = utils.loadCourses("../assets/courses.json")
    env = CourseRecEnv(courses)

    profile = {"major": "CS", "year": 3}
    state = env.reset(profile)

    # model structure and load weights
    state_dim = 4
    action_size = len(courses)
    model = DQN(state_dim, action_size)
    model.load_state_dict(torch.load("dqn_model.pth"))
    model.eval()

    print(f"\n🔎 Recommendations for {profile['major']} Year {profile['year']} student:\n")

    for i in range(5):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        course = courses[action]
        title = f"{course['subject']} {course['number']}: {course['title']}"
        print(f"Recommended: {title}")
        state, _, _, _ = env.step(action)

        hovered_titles = [f"{courses[idx]['subject']} {courses[idx]['number']}: {courses[idx]['title']}" for idx in env.current_state["hovered_courses"]]
        watchlist_titles = [f"{courses[idx]['subject']} {courses[idx]['number']}: {courses[idx]['title']}" for idx in env.current_state["watchlist"]]

        print(f"  Hovered courses: {hovered_titles}")
        print(f"  Watchlist: {watchlist_titles}\n")

if __name__ == "__main__":
    runSession()
