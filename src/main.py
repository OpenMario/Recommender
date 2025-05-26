import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import json


class ClassificationLevel(Enum):
    FRESHMAN = 0
    SOPHOMORE = 1
    PRE_JUNIOR = 2
    JUNIOR = 3
    SENIOR = 4


class StudentContext(BaseModel):
    """Represents the context/state of a student"""
    major: str
    classification: ClassificationLevel
    previous_courses: List[str] = Field(
        default_factory=list, description="List of subject codes (e.g., ['MATH', 'PHYS'])")
    current_gpa: float = Field(ge=0.0, le=4.0, description="GPA on 4.0 scale")
    credits_completed: int = Field(ge=0, description="Total credits completed")
    time_preference: str = Field(
        description="'morning', 'afternoon', 'evening'")
    preferred_days: List[str] = Field(
        default_factory=list, description="['Monday', 'Tuesday', etc.]")
    current_search_text: str = Field(
        default="", description="Current search query")
    active_filters: List[str] = Field(
        default_factory=list, description="Active search filters")
    session_hovers: List[str] = Field(
        default_factory=list, description="Courses hovered in current session")
    session_watchlisted: List[str] = Field(
        default_factory=list, description="Courses added to watchlist")
    session_paginations: int = Field(
        default=0, ge=0, description="Number of times user went to next page")

    @field_validator('time_preference')
    @classmethod
    def validate_time_preference(cls, v):
        valid_times = ['morning', 'afternoon', 'evening']
        if v not in valid_times:
            raise ValueError(f'time_preference must be one of {valid_times}')
        return v

    @field_validator('preferred_days')
    @classmethod
    def validate_preferred_days(cls, v):
        valid_days = ['Monday', 'Tuesday', 'Wednesday',
                      'Thursday', 'Friday', 'Saturday', 'Sunday']
        invalid_days = [day for day in v if day not in valid_days]
        if invalid_days:
            raise ValueError(f'Invalid days: {
                             invalid_days}. Must be from {valid_days}')
        return v

    class Config:
        use_enum_values = True


class CourseAction(BaseModel):
    """Represents a course recommendation action"""
    course_subject: str
    course_number: str
    course_title: str
    credits: str
    college_code: str
    section_info: Dict = Field(
        default_factory=dict, description="Contains CRN, times, days, etc.")

    class Config:
        arbitrary_types_allowed = True


class CourseRecommendationBandit:
    """
    Contextual bandit for course recommendations using student context
    and behavioral signals.
    """

    def __init__(self, available_courses: List[Dict], learning_rate: float = 0.01,
                 epsilon: float = 0.15, epsilon_decay: float = 0.995):
        """
        Initialize the course recommendation bandit.

        Args:
            available_courses: List of available course dictionaries
            learning_rate: Learning rate for weight updates
            epsilon: Exploration probability
            epsilon_decay: Decay factor for epsilon
        """
        self.available_courses = available_courses
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Define context feature dimensions
        self.context_dim = self._calculate_context_dimension()

        # We'll treat each unique course as an action
        self.course_actions = self._create_course_actions()
        self.n_actions = len(self.course_actions)

        # Initialize weights for each course action
        self.weights = np.random.normal(
            0, 0.01, (self.n_actions, self.context_dim))

        # Track statistics
        self.total_reward = 0
        self.n_steps = 0
        self.reward_history = []
        self.action_counts = np.zeros(self.n_actions)

        # Subject mappings for encoding
        self.subject_to_idx = {subj: idx for idx,
                               subj in enumerate(self._get_all_subjects())}
        self.major_to_idx = {maj: idx for idx,
                             maj in enumerate(self._get_common_majors())}

    def _calculate_context_dimension(self) -> int:
        """Calculate the dimension of the context vector"""
        # Student features:
        # - Major (one-hot encoded, ~50 common majors)
        # - Classification level (5 levels)
        # - GPA (1 continuous)
        # - Credits completed (1 continuous, normalized)
        # - Time preference (3 categories: morning/afternoon/evening)
        # - Day preferences (7 binary features for each day)
        # - Previous courses (binary vector for each subject, ~160 subjects)
        # - Current search features (TF-IDF style features, ~20)
        # - Active filters (binary vector, ~10 common filters)
        # - Behavioral signals:
        #   - Session hovers count (1)
        #   - Session watchlist count (1)
        #   - Session pagination count (1)

        major_dim = 50  # Common majors
        classification_dim = 5
        continuous_dim = 2  # GPA + credits
        time_pref_dim = 3
        day_pref_dim = 7
        subject_history_dim = 160  # All subjects
        search_features_dim = 20
        filter_dim = 10
        behavioral_dim = 3

        return (major_dim + classification_dim + continuous_dim +
                time_pref_dim + day_pref_dim + subject_history_dim +
                search_features_dim + filter_dim + behavioral_dim)

    def _get_all_subjects(self) -> List[str]:
        """Get list of all subject codes"""
        from subjects import subjects
        return subjects

    def _get_common_majors(self) -> List[str]:
        """Get list of common majors"""
        from majors import majors
        return majors

    def _create_course_actions(self) -> List[CourseAction]:
        """Create list of possible course actions from available courses"""
        actions = []

        # Create actions from actual course data
        # Limit to first 100 for performance
        for course in self.available_courses[:100]:
            actions.append(CourseAction(
                course_subject=course.get('subject_id', ''),
                course_number=course.get('course_number', ''),
                course_title=course.get('title', ''),
                credits=str(course.get('credits', '3')),
                college_code=course.get('college_id', ''),
                section_info={
                    'crn': course.get('crn', ''),
                    'section': course.get('section', ''),
                    'days': course.get('days', []),
                    'start_time': course.get('start_time', ''),
                    'end_time': course.get('end_time', ''),
                    'instructors': course.get('instructors', [])
                }
            ))

        return actions

    def _encode_context(self, context: StudentContext) -> np.ndarray:
        """
        Encode student context into feature vector.

        Args:
            context: Student context object

        Returns:
            Encoded context vector
        """
        features = []

        # Major (one-hot)
        major_vec = np.zeros(50)
        if context.major in self.major_to_idx:
            major_vec[self.major_to_idx[context.major]] = 1
        features.extend(major_vec)

        # Classification level (one-hot)
        classification_vec = np.zeros(5)
        # Handle both enum instances and integer values
        if isinstance(context.classification, ClassificationLevel):
            classification_idx = context.classification.value
        else:
            classification_idx = int(context.classification)
        classification_vec[classification_idx] = 1
        features.extend(classification_vec)

        # Continuous features (normalized)
        features.append(context.current_gpa / 4.0)  # Normalize GPA
        # Normalize credits
        features.append(min(context.credits_completed / 150.0, 1.0))

        # Time preference (one-hot)
        time_prefs = ['morning', 'afternoon', 'evening']
        time_vec = np.zeros(3)
        if context.time_preference in time_prefs:
            time_vec[time_prefs.index(context.time_preference)] = 1
        features.extend(time_vec)

        # Day preferences (binary)
        all_days = ['Monday', 'Tuesday', 'Wednesday',
                    'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_vec = [1 if day in context.preferred_days else 0 for day in all_days]
        features.extend(day_vec)

        # Previous courses (binary vector for subjects)
        all_subjects = self._get_all_subjects()
        subject_vec = np.zeros(len(all_subjects))
        for course in context.previous_courses:
            if course in self.subject_to_idx:
                subject_vec[self.subject_to_idx[course]] = 1

        # Pad or truncate to expected size
        if len(subject_vec) < 160:
            subject_vec = np.pad(subject_vec, (0, 160 - len(subject_vec)))
        else:
            subject_vec = subject_vec[:160]
        features.extend(subject_vec)

        # Search features (simplified TF-IDF style)
        search_vec = np.zeros(20)
        if context.current_search_text:
            # Simple hash-based features for search text
            for i, word in enumerate(context.current_search_text.lower().split()[:20]):
                # Simple word importance
                search_vec[i % 20] += len(word) / 10.0
        features.extend(search_vec)

        # Active filters (binary)
        common_filters = ['morning', 'afternoon', 'evening', 'monday', 'tuesday',
                          'wednesday', 'thursday', 'friday', 'online', 'in-person']
        filter_vec = [
            1 if f in context.active_filters else 0 for f in common_filters]
        features.extend(filter_vec)

        # Behavioral signals
        # Normalize hover count
        features.append(min(len(context.session_hovers) / 10.0, 1.0))
        # Normalize watchlist
        features.append(min(len(context.session_watchlisted) / 5.0, 1.0))
        # Normalize pagination
        features.append(min(context.session_paginations / 10.0, 1.0))

        # Ensure correct dimension
        feature_array = np.array(features)
        if len(feature_array) < self.context_dim:
            feature_array = np.pad(
                feature_array, (0, self.context_dim - len(feature_array)))
        else:
            feature_array = feature_array[:self.context_dim]

        return feature_array

    def predict_rewards(self, context: StudentContext) -> np.ndarray:
        """Predict expected rewards for all course actions given context."""
        context_vec = self._encode_context(context)
        return np.dot(self.weights, context_vec)

    def select_courses(self, context: StudentContext, n_recommendations: int = 5) -> List[int]:
        """
        Select top N course recommendations using epsilon-greedy policy.

        Args:
            context: Student context
            n_recommendations: Number of courses to recommend

        Returns:
            List of course action indices
        """
        n_recommendations = min(n_recommendations, self.n_actions)

        if np.random.random() < self.epsilon:
            # Explore: random courses
            return np.random.choice(self.n_actions, size=n_recommendations, replace=False).tolist()
        else:
            # Exploit: top predicted courses
            predicted_rewards = self.predict_rewards(context)
            return np.argsort(predicted_rewards)[-n_recommendations:].tolist()

    def update(self, context: StudentContext, recommended_courses: List[int],
               rewards: List[float]):
        """
        Update the model based on observed rewards.

        Args:
            context: Student context
            recommended_courses: List of course indices that were recommended
            rewards: List of rewards for each recommended course
        """
        context_vec = self._encode_context(context)

        for course_idx, reward in zip(recommended_courses, rewards):
            # Predict current reward for the course
            predicted_reward = np.dot(self.weights[course_idx], context_vec)

            # Calculate prediction error
            error = reward - predicted_reward

            # Update weights using gradient descent
            self.weights[course_idx] += self.learning_rate * \
                error * context_vec

            # Update statistics
            self.total_reward += reward
            self.action_counts[course_idx] += 1

        self.n_steps += 1
        self.reward_history.extend(rewards)

        # Decay epsilon
        self.epsilon *= self.epsilon_decay

    def calculate_reward(self, context: StudentContext, course_idx: int,
                         user_action: str) -> float:
        """
        Calculate reward based on user interaction with recommended course.

        Args:
            context: Student context
            course_idx: Index of recommended course
            user_action: Type of user action ('hover', 'watchlist', 'click', 'enroll', 'pagination')

        Returns:
            Calculated reward
        """
        base_reward = 0.0
        course = self.course_actions[course_idx]

        # Positive rewards
        if user_action == 'hover':
            base_reward = 0.1  # Small positive signal
        elif user_action == 'watchlist':
            base_reward = 0.5  # Strong positive signal
        elif user_action == 'click':
            base_reward = 0.3  # Moderate positive signal

        # Negative rewards
        elif user_action == 'pagination':
            base_reward = -0.1  # User went to next page without interacting

        # Contextual adjustments
        if course.course_subject in context.previous_courses:
            base_reward *= 0.8  # Slight penalty for recommending same subject

        if course.course_subject.lower() in context.current_search_text.lower():
            base_reward *= 1.2  # Bonus for matching search intent

        return base_reward

    def get_recommendation_explanation(self, context: StudentContext,
                                       course_idx: int) -> str:
        """Generate explanation for why a course was recommended."""
        course = self.course_actions[course_idx]
        context_vec = self._encode_context(context)
        predicted_reward = np.dot(self.weights[course_idx], context_vec)

        explanations = []

        if course.course_subject in context.previous_courses:
            explanations.append(f"builds on your experience with {
                                course.course_subject}")

        if course.course_subject.lower() in context.current_search_text.lower():
            explanations.append("matches your search criteria")

        if isinstance(context.classification, ClassificationLevel):
            classification = context.classification
        else:
            classification = ClassificationLevel(context.classification)

        if classification in [ClassificationLevel.FRESHMAN, ClassificationLevel.SOPHOMORE]:
            explanations.append("suitable for your academic level")

        if predicted_reward > 0.5:
            explanations.append("highly recommended based on your profile")

        if not explanations:
            explanations.append("recommended based on your academic profile")

        return f"This course is {' and '.join(explanations)}."


def create_sample_context() -> StudentContext:
    """Create a sample student context for testing."""
    return StudentContext(
        major="Computer Science (BS)",
        classification=ClassificationLevel.SOPHOMORE,
        previous_courses=["MATH", "PHYS", "ENGL"],
        current_gpa=3.2,
        credits_completed=45,
        time_preference="morning",
        preferred_days=["Monday", "Wednesday", "Friday"],
        current_search_text="programming algorithms",
        active_filters=["morning", "in-person"],
        session_hovers=["CS", "MATH"],
        session_watchlisted=["CS"],
        session_paginations=2
    )


# Example usage
if __name__ == "__main__":
    from section import courses
    bandit = CourseRecommendationBandit(courses)

    # Create sample context
    student_context = create_sample_context()

    # Get recommendations
    recommended_courses = bandit.select_courses(
        student_context, n_recommendations=2)

    print("Course Recommendations:")
    for i, course_idx in enumerate(recommended_courses):
        course = bandit.course_actions[course_idx]
        explanation = bandit.get_recommendation_explanation(
            student_context, course_idx)
        print(
            f"{i+1}. {course.course_subject} {course.course_number}: {course.course_title}")
        print(f"   {explanation}")

    # Simulate user interactions and update
    rewards = []
    for course_idx in recommended_courses:
        # Simulate different user actions
        action = np.random.choice(['hover', 'watchlist', 'click', 'pagination'],
                                  p=[0.4, 0.2, 0.3, 0.1])
        reward = bandit.calculate_reward(student_context, course_idx, action)
        rewards.append(reward)
        print(f"User {action} on {
              bandit.course_actions[course_idx].course_subject}: reward = {reward}")

    # Update bandit
    bandit.update(student_context, recommended_courses, rewards)

    print(f"\nBandit updated. Current epsilon: {bandit.epsilon:.3f}")
    print(f"Average reward: {np.mean(bandit.reward_history):.3f}")
