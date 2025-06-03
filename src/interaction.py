import numpy as np
import random
from typing import List, Dict, Tuple, Generator
from dataclasses import dataclass
from enum import Enum
import json


from model import (
    StudentContext, ClassificationLevel, CourseRecommendationBandit,
    CourseAction, create_sample_context
)


class InteractionType(Enum):
    """Types of user interactions with course recommendations"""
    HOVER = "hover"
    CLICK = "click"
    WATCHLIST = "watchlist"
    IGNORE = "ignore"
    PAGINATION = "pagination"
    SEARCH_REFINEMENT = "search_refinement"
    FILTER_CHANGE = "filter_change"


@dataclass
class UserInteraction:
    """Represents a single user interaction with the recommendation system"""
    session_id: str
    student_context: StudentContext
    recommended_courses: List[int]
    interaction_type: InteractionType

    course_idx: int
    timestamp: float
    reward: float
    metadata: Dict = None


class InteractionGenerator:
    """
    Generates realistic user interactions with course recommendations
    based on student context and course characteristics.
    """

    def __init__(self, bandit: CourseRecommendationBandit, seed: int = 42):
        self.bandit = bandit
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

        self.base_interaction_probs = {
            InteractionType.HOVER: 0.6,
            InteractionType.CLICK: 0.25,
            InteractionType.WATCHLIST: 0.15,
            InteractionType.IGNORE: 0.3,
            InteractionType.PAGINATION: 0.1,
            InteractionType.SEARCH_REFINEMENT: 0.05,
            InteractionType.FILTER_CHANGE: 0.03
        }

        self.gpa_multipliers = {
            "high": 1.3,
            "medium": 1.0,
            "low": 0.7
        }

        self.classification_multipliers = {
            ClassificationLevel.FRESHMAN: 1.2,
            ClassificationLevel.SOPHOMORE: 1.1,
            ClassificationLevel.PRE_JUNIOR: 1.0,
            ClassificationLevel.JUNIOR: 0.9,
            ClassificationLevel.SENIOR: 0.8
        }

    def _get_gpa_category(self, gpa: float) -> str:
        """Categorize GPA into high/medium/low"""
        if gpa >= 3.5:
            return "high"
        elif gpa >= 2.5:
            return "medium"
        else:
            return "low"

    def _calculate_interaction_probability(self, context: StudentContext,
                                           course: CourseAction,
                                           interaction_type: InteractionType) -> float:
        """
        Calculate probability of a specific interaction type based on context and course.
        """
        base_prob = self.base_interaction_probs[interaction_type]

        gpa_category = self._get_gpa_category(context.current_gpa)
        gpa_multiplier = self.gpa_multipliers[gpa_category]

        if isinstance(context.classification, ClassificationLevel):
            classification = context.classification
        else:
            classification = ClassificationLevel(context.classification)

        class_multiplier = self.classification_multipliers[classification]

        relevance_multiplier = 1.0

        if course.course_subject.lower() in context.current_search_text.lower():
            relevance_multiplier *= 1.5

        if course.course_subject in context.previous_courses:
            relevance_multiplier *= 1.3

        session_activity = len(context.session_hovers) + \
            len(context.session_watchlisted)
        activity_multiplier = 1.0 + (session_activity * 0.1)

        if interaction_type == InteractionType.WATCHLIST:

            if len(context.session_watchlisted) > 3:
                relevance_multiplier *= 0.5

        elif interaction_type == InteractionType.PAGINATION:

            if session_activity == 0:
                relevance_multiplier *= 2.0

        elif interaction_type == InteractionType.HOVER:

            if gpa_category == "high":
                relevance_multiplier *= 1.2

        final_prob = base_prob * gpa_multiplier * class_multiplier * \
            relevance_multiplier * activity_multiplier
        return min(final_prob, 1.0)

    def _generate_single_interaction(self, context: StudentContext,
                                     recommended_courses: List[int],
                                     session_id: str,
                                     timestamp: float) -> UserInteraction:
        """Generate a single interaction for a recommendation session."""

        interaction_probs = {}
        courses = [self.bandit.course_actions[idx]
                   for idx in recommended_courses]

        for interaction_type in InteractionType:
            if interaction_type in [InteractionType.PAGINATION,
                                    InteractionType.SEARCH_REFINEMENT,
                                    InteractionType.FILTER_CHANGE]:

                prob = self._calculate_interaction_probability(
                    context, courses[0], interaction_type)
                interaction_probs[interaction_type] = prob
            else:

                avg_prob = np.mean([
                    self._calculate_interaction_probability(
                        context, course, interaction_type)
                    for course in courses
                ])
                interaction_probs[interaction_type] = avg_prob

        interaction_types = list(interaction_probs.keys())
        probabilities = list(interaction_probs.values())

        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(probabilities)] * len(probabilities)

        selected_interaction = self.rng.choice(
            interaction_types, p=probabilities)

        course_idx = -1
        if selected_interaction not in [InteractionType.PAGINATION,
                                        InteractionType.SEARCH_REFINEMENT,
                                        InteractionType.FILTER_CHANGE]:

            course_probs = [
                self._calculate_interaction_probability(
                    context, course, selected_interaction)
                for course in courses
            ]

            if sum(course_probs) > 0:
                course_probs = np.array(course_probs) / sum(course_probs)
                selected_course_pos = self.rng.choice(
                    len(recommended_courses), p=course_probs)
                course_idx = recommended_courses[selected_course_pos]
            else:
                course_idx = self.rng.choice(recommended_courses)

        if course_idx != -1:
            reward = self.bandit.calculate_reward(
                context, course_idx, selected_interaction.value)
        else:

            if selected_interaction == InteractionType.PAGINATION:
                reward = -0.1
            elif selected_interaction == InteractionType.SEARCH_REFINEMENT:
                reward = 0.05
            elif selected_interaction == InteractionType.FILTER_CHANGE:
                reward = 0.02
            else:
                reward = 0.0

        metadata = {
            "interaction_probabilities": interaction_probs,
            "gpa_category": self._get_gpa_category(context.current_gpa),
            "session_activity_level": len(context.session_hovers) + len(context.session_watchlisted)
        }

        if course_idx != -1:
            course = self.bandit.course_actions[course_idx]
            metadata["course_subject"] = course.course_subject
            metadata["course_number"] = course.course_number
            metadata["course_title"] = course.course_title

        return UserInteraction(
            session_id=session_id,
            student_context=context,
            recommended_courses=recommended_courses,
            interaction_type=selected_interaction,
            course_idx=course_idx,
            timestamp=timestamp,
            reward=reward,
            metadata=metadata
        )

    def generate_session_interactions(self, context: StudentContext,
                                      n_recommendations: int = 5,
                                      max_interactions: int = 10,
                                      session_id: str = None) -> List[UserInteraction]:
        """
        Generate a sequence of interactions for a single recommendation session.
        """
        if session_id is None:
            session_id = f"session_{self.rng.randint(10000, 99999)}"

        recommended_courses = self.bandit.select_courses(
            context, n_recommendations)

        interactions = []
        timestamp = 0.0

        for i in range(max_interactions):
            interaction = self._generate_single_interaction(
                context, recommended_courses, session_id, timestamp
            )
            interactions.append(interaction)

            context = self._update_context_from_interaction(
                context, interaction)

            if interaction.interaction_type == InteractionType.PAGINATION:
                if self.rng.random() < 0.3:
                    break
            elif interaction.interaction_type == InteractionType.WATCHLIST:
                if self.rng.random() < 0.4:
                    break
            elif len(interactions) >= 3 and self.rng.random() < 0.2:

                break

            timestamp += self.rng.exponential(30.0)

        return interactions

    def _update_context_from_interaction(self, context: StudentContext,
                                         interaction: UserInteraction) -> StudentContext:
        """Update student context based on their interaction."""

        new_context = StudentContext(
            major=context.major,
            classification=context.classification,
            previous_courses=context.previous_courses.copy(),
            current_gpa=context.current_gpa,
            credits_completed=context.credits_completed,
            time_preference=context.time_preference,
            preferred_days=context.preferred_days.copy(),
            current_search_text=context.current_search_text,
            active_filters=context.active_filters.copy(),
            session_hovers=context.session_hovers.copy(),
            session_watchlisted=context.session_watchlisted.copy(),
            session_paginations=context.session_paginations
        )

        if interaction.interaction_type == InteractionType.HOVER and interaction.course_idx != -1:
            course = self.bandit.course_actions[interaction.course_idx]
            if course.course_subject not in new_context.session_hovers:
                new_context.session_hovers.append(course.course_subject)

        elif interaction.interaction_type == InteractionType.WATCHLIST and interaction.course_idx != -1:
            course = self.bandit.course_actions[interaction.course_idx]
            if course.course_subject not in new_context.session_watchlisted:
                new_context.session_watchlisted.append(course.course_subject)

        elif interaction.interaction_type == InteractionType.PAGINATION:
            new_context.session_paginations += 1

        elif interaction.interaction_type == InteractionType.SEARCH_REFINEMENT:

            search_terms = ["advanced", "intro",
                            "programming", "theory", "lab", "online"]
            new_term = self.rng.choice(search_terms)
            new_context.current_search_text = f"{
                context.current_search_text} {new_term}".strip()

        elif interaction.interaction_type == InteractionType.FILTER_CHANGE:

            available_filters = ["morning", "afternoon",
                                 "evening", "online", "in-person"]
            if self.rng.random() < 0.5 and new_context.active_filters:

                filter_to_remove = self.rng.choice(new_context.active_filters)
                new_context.active_filters.remove(filter_to_remove)
            else:

                available = [
                    f for f in available_filters if f not in new_context.active_filters]
                if available:
                    new_filter = self.rng.choice(available)
                    new_context.active_filters.append(new_filter)

        return new_context

    def generate_interactions(self, n_sessions: int = 100,
                              students_pool: List[StudentContext] = None) -> Generator[UserInteraction, None, None]:
        """
        Generate a stream of interactions across multiple sessions and students.

        Args:
            n_sessions: Number of sessions to generate
            students_pool: Pool of student contexts to sample from

        Yields:
            Individual UserInteraction objects
        """
        if students_pool is None:

            students_pool = self._create_diverse_student_pool(50)

        for session_num in range(n_sessions):

            student = self.rng.choice(students_pool)

            session_interactions = self.generate_session_interactions(
                student,
                session_id=f"session_{session_num:04d}"
            )

            for interaction in session_interactions:
                yield interaction

    def _create_diverse_student_pool(self, n_students: int) -> List[StudentContext]:
        """Create a diverse pool of student contexts for testing."""
        students = []

        majors = ["Computer Science (BS)", "Mathematics (BS)", "Physics (BS)",
                  "Engineering (BS)", "Biology (BS)", "Chemistry (BS)", "English (BA)"]

        for i in range(n_students):

            course_options = [
                ["MATH", "PHYS"],
                ["ENGL", "HIST"],
                ["CHEM", "BIOL"],
                ["CS", "MATH"],
                ["ENGR", "PHYS"]
            ]
            selected_courses_set = course_options[self.rng.randint(
                0, len(course_options))]
            n_courses = self.rng.randint(
                1, min(4, len(selected_courses_set) + 1))
            previous_courses = self.rng.choice(selected_courses_set, size=min(
                n_courses, len(selected_courses_set)), replace=False).tolist()

            day_options = [
                ["Monday", "Wednesday", "Friday"],
                ["Tuesday", "Thursday"],
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            ]
            selected_days = day_options[self.rng.randint(0, len(day_options))]

            filter_options = [[], ["morning"], [
                "online"], ["morning", "online"]]
            selected_filters = filter_options[self.rng.randint(
                0, len(filter_options))]

            students.append(StudentContext(
                major=self.rng.choice(majors),
                classification=ClassificationLevel(self.rng.randint(0, 5)),
                previous_courses=previous_courses,

                current_gpa=np.clip(self.rng.normal(3.0, 0.5), 0.0, 4.0),
                credits_completed=self.rng.randint(0, 120),
                time_preference=self.rng.choice(
                    ["morning", "afternoon", "evening"]),
                preferred_days=selected_days,
                current_search_text=self.rng.choice([
                    "programming", "calculus", "physics", "chemistry",
                    "literature", "history", "statistics"
                ]),
                active_filters=selected_filters,
                session_hovers=[],
                session_watchlisted=[],
                session_paginations=0
            ))

        return students


def demo_interaction_generator():
    """Demonstrate the interaction generator."""

    print("Course Recommendation Interaction Generator Demo")
    print("=" * 50)

    print("Interaction Types Available:")
    for interaction_type in InteractionType:
        print(f"  - {interaction_type.value}")

    print("\nSample interaction data structure:")
    sample_context = create_sample_context()
    print(f"Student: {sample_context.major}, GPA: {
          sample_context.current_gpa}")
    print(f"Search: '{sample_context.current_search_text}'")

    sample_interaction = UserInteraction(
        session_id="demo_session",
        student_context=sample_context,
        recommended_courses=[0, 1, 2],
        interaction_type=InteractionType.HOVER,
        course_idx=0,
        timestamp=10.5,
        reward=0.1,
        metadata={"course_subject": "CS", "gpa_category": "medium"}
    )

    print(f"\nSample Interaction:")
    print(f"  Type: {sample_interaction.interaction_type.value}")
    print(f"  Course Index: {sample_interaction.course_idx}")
    print(f"  Reward: {sample_interaction.reward}")
    print(f"  Timestamp: {sample_interaction.timestamp}")


if __name__ == "__main__":
    demo_interaction_generator()
