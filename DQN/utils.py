import os
import json

def encode_major(major):
    """
    Encodes major string into an integer ID.
    Extend the dictionary as needed.
    """
    major_map = {
        "CS": 0,
        "ME": 1,
        "BIO": 2,
        "PSY": 3,
        "EE": 4,
        "DSCI": 5,
        "SE": 6,
        "CHE": 7,
        "MATH": 8
    }
    return major_map.get(major.upper(), len(major_map))  # unknown majors go last

def loadCourses(json_path):
    """
    Loads and flattens courses from JSON into a list of dicts with fields:
    - subject, number, title, description, credits
    """
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
