import json
import random
import utils

class CourseRecEnv:
  def __init__(self, courses):
    self.courses = courses
    self.current_state = None

  def reset(self, profile):
    self.current_state = {
      "profile": utils.encodeProfile(profile),
      "hovered_courses": set(),
      "watchlist": set()
    }
    return self._getState()

  def _getState(self):
    hover_count = len(self.current_state["hovered_courses"])
    watchlist_count = len(self.current_state["watchlist"])
    return self.current_state["profile"] + (hover_count, watchlist_count)

  def step(self, action):
    reward = 0
    done = False
    info = {}

    if action == 1:  # Recommend courses to explore
      recommended = random.sample(self.courses, min(3, len(self.courses)))
      hovered = random.choice(recommended)
      course_id = f"{hovered['subject']} {hovered['number']}"
      self.current_state["hovered_courses"].add(course_id)
      reward += 1
      if random.random() < 0.3:
        self.current_state["watchlist"].add(course_id)
        reward += 5

    elif action == 2:  # Recommend filters
      #TODO:elaborate action 
      reward += 0.5

    elif action == 3:  # Recommend search terms
      #TODO:elaborate action
      reward += 0.3

    return self._getState(), reward, done, info

