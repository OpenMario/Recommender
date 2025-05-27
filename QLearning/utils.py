import os
import json
import pickle

def is_ai_course(description):
    keywords = ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'AI']
    description_lower = description.lower()
    return any(keyword in description_lower for keyword in keywords)

def is_software_engineering_course(description):
    keywords = ['software engineering', 'software development', 'agile', 'devops', 'version control']
    description_lower = description.lower()
    return any(keyword in description_lower for keyword in keywords)

def loadCourses(json_path):
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, json_path)

    with open(full_path, 'r') as f:
        data = json.load(f)

    courses = []
    for subject_code, subject_data in data.items():
        if "courses" in subject_data:
            for course in subject_data["courses"]:
                courses.append({
                    "subject": subject_code,
                    "number": course.get("number", ""),
                    "title": course.get("title", ""),
                    "description": course.get("description", ""),
                    "credits": course.get("credits", 0)
                })
    return courses

def encodeProfile(profile):
  major_map = {'CS': 0, 'SE': 1, 'DSCI': 2}
  major_vec = [0] * len(major_map)
  major_idx = major_map.get(profile.get('major', 'CS'), 0)
  major_vec[major_idx] = 1

  year_norm = profile.get('year', 1) / 5

  interest_keywords = ['AI', 'SE', 'Algorithms', 'Game', 'THEORY']
  interest_vec = [1 if topic in profile.get('interests', []) else 0 for topic in interest_keywords]

  return tuple(major_vec + [year_norm] + interest_vec)

def readPickle(pickle_file, json_file=''):
  with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

  print(data)

  #with open('data.json', 'w') as f:
    #json.dump(data, f, indent=2)

readPickle('trained_agent.pkl')