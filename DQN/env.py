
import json
import random
import utils
import numpy as np

class CourseRecEnv:
    def __init__(self, courses):
        self.courses = courses
        self.current_state = None

    def reset(self, profile):
        self.profile = profile
        self.current_state = {
            "hovered_courses": set(),
            "watchlist": set(),
            "last_hovered": -1,
            "last_watchlisted": -1
        }
        return self._getState()

    def _getState(self):
        """
        Returns a numerical state vector:
        [major_encoding, year, last_hovered_id, last_watchlisted_id]
        All values are normalized to [0, 1] range if possible.
        """
        major_encoding = utils.encode_major(self.profile["major"]) / 10.0  # normalize
        year = self.profile["year"] / 5.0  # assuming max year = 5
        hovered = self.current_state["last_hovered"] / len(self.courses) if self.current_state["last_hovered"] >= 0 else 0.0
        watchlisted = self.current_state["last_watchlisted"] / len(self.courses) if self.current_state["last_watchlisted"] >= 0 else 0.0
        return np.array([major_encoding, year, hovered, watchlisted], dtype=np.float32)

    def step(self, action):
        course = self.courses[action]
        reward = 0

        course_id = action
        self.current_state["hovered_courses"].add(course_id)
        self.current_state["last_hovered"] = course_id
        reward += 1

        if course.get("subject", "").lower() == self.profile["major"].lower():
            reward += 3

        watchlist_prob = 0.3
        subject = course.get("subject", "").lower()
        number_str = course.get("number", "100")
        try:
            number = int(number_str)
        except ValueError:
            number = 100

        if subject == self.profile["major"].lower():
            watchlist_prob = 0.6

        year = self.profile.get("year", 1)
        if (year == 1 and number < 300) or \
           (year == 2 and number < 400) or \
           (year == 3 and number < 500) or \
           (year >= 4):
            watchlist_prob += 0.2
        else:
            watchlist_prob -= 0.2

        watchlist_prob = max(0.0, min(watchlist_prob, 1.0))

        if random.random() < watchlist_prob:
            self.current_state["watchlist"].add(course_id)
            self.current_state["last_watchlisted"] = course_id
            reward += 5

        new_state = self._getState()
        return new_state, reward, False, {}

#print("[DEBUG] env.py loaded.")

