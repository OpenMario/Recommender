
"""
Complete Bandit Evaluation Main Script

This script demonstrates comprehensive evaluation of the course recommendation
contextual bandit system using all evaluation methods except live data.

Includes:
- Simulation-based evaluation
- Offline evaluation with historical data
- Cross-validation
- A/B testing between different bandit configurations
- Performance analysis and reporting
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import seaborn as sns


from model import (
    StudentContext, CourseRecommendationBandit,
    ClassificationLevel, create_sample_context
)
from interaction import InteractionGenerator, UserInteraction, InteractionType
from eval import (
    ComprehensiveEvaluationSuite, OnlineEvaluator, OfflineEvaluator,
    SimulationEvaluator, CrossValidationEvaluator, EvaluationMetrics
)


MOCK_COURSES = [
    {
        'subject_id': 'CS', 'course_number': '101', 'title': 'Intro to Programming',
        'credits': 3, 'college_id': 'ENGR', 'crn': '12345', 'section': '001',
        'days': ['Monday', 'Wednesday', 'Friday'], 'start_time': '09:00',
        'end_time': '10:00', 'instructors': ['Dr. Smith']
    },
    {
        'subject_id': 'CS', 'course_number': '201', 'title': 'Data Structures',
        'credits': 3, 'college_id': 'ENGR', 'crn': '12346', 'section': '001',
        'days': ['Tuesday', 'Thursday'], 'start_time': '14:00',
        'end_time': '15:30', 'instructors': ['Dr. Johnson']
    },
    {
        'subject_id': 'MATH', 'course_number': '151', 'title': 'Calculus I',
        'credits': 4, 'college_id': 'ARTS', 'crn': '12347', 'section': '001',
        'days': ['Monday', 'Wednesday', 'Friday'], 'start_time': '11:00',
        'end_time': '12:00', 'instructors': ['Dr. Brown']
    },
    {
        'subject_id': 'PHYS', 'course_number': '201', 'title': 'Physics I',
        'credits': 4, 'college_id': 'ARTS', 'crn': '12348', 'section': '001',
        'days': ['Tuesday', 'Thursday'], 'start_time': '10:00',
        'end_time': '11:30', 'instructors': ['Dr. Wilson']
    },
    {
        'subject_id': 'ENGL', 'course_number': '101', 'title': 'Composition I',
        'credits': 3, 'college_id': 'ARTS', 'crn': '12349', 'section': '001',
        'days': ['Monday', 'Wednesday'], 'start_time': '13:00',
        'end_time': '14:30', 'instructors': ['Prof. Davis']
    }
] * 20


def create_diverse_student_contexts(n_students: int = 100) -> List[StudentContext]:
    """Create a diverse set of student contexts for testing"""
    np.random.seed(42)
    students = []

    majors = [
        "Computer Science (BS)", "Mathematics (BS)", "Physics (BS)",
        "Engineering (BS)", "Biology (BS)", "Chemistry (BS)",
        "English (BA)", "Psychology (BS)", "Business (BS)"
    ]

    subjects_by_major = {
        "Computer Science (BS)": ["CS", "MATH", "ENGR"],
        "Mathematics (BS)": ["MATH", "PHYS", "CS"],
        "Physics (BS)": ["PHYS", "MATH", "ENGR"],
        "Engineering (BS)": ["ENGR", "MATH", "PHYS"],
        "Biology (BS)": ["BIOL", "CHEM", "MATH"],
        "Chemistry (BS)": ["CHEM", "MATH", "PHYS"],
        "English (BA)": ["ENGL", "HIST", "PHIL"],
        "Psychology (BS)": ["PSYC", "MATH", "BIOL"],
        "Business (BS)": ["BUSI", "MATH", "ECON"]
    }

    search_terms_by_major = {
        "Computer Science (BS)": ["programming", "algorithms", "software", "coding"],
        "Mathematics (BS)": ["calculus", "algebra", "statistics", "analysis"],
        "Physics (BS)": ["mechanics", "thermodynamics", "quantum", "waves"],
        "Engineering (BS)": ["circuits", "mechanics", "design", "materials"],
        "Biology (BS)": ["genetics", "ecology", "molecular", "anatomy"],
        "Chemistry (BS)": ["organic", "inorganic", "analytical", "physical"],
        "English (BA)": ["literature", "writing", "poetry", "rhetoric"],
        "Psychology (BS)": ["cognitive", "behavioral", "development", "research"],
        "Business (BS)": ["management", "finance", "marketing", "economics"]
    }

    for i in range(n_students):
        major = np.random.choice(majors)
        classification = ClassificationLevel(np.random.randint(0, 5))

        relevant_subjects = subjects_by_major.get(major, ["MATH", "ENGL"])
        n_prev_courses = min(classification.value * 2, len(relevant_subjects))
        previous_courses = np.random.choice(
            relevant_subjects,
            size=n_prev_courses,
            replace=False
        ).tolist() if n_prev_courses > 0 else []

        base_gpa = 2.5 + classification.value * 0.2
        gpa = np.clip(np.random.normal(base_gpa, 0.4),
                      0.0, 4.0)

        base_credits = classification.value * 25
        credits = base_credits + np.random.randint(-10, 15)

        search_options = search_terms_by_major.get(major, ["general"])
        search_term = np.random.choice(search_options)

        day_options = [
            ["Monday", "Wednesday", "Friday"],
            ["Tuesday", "Thursday"],
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        ]
        selected_days = day_options[np.random.randint(0, len(day_options))]

        filter_options = [
            [], ["morning"], ["online"], ["morning", "online"]
        ]
        selected_filters = filter_options[np.random.randint(
            0, len(filter_options))]

        students.append(StudentContext(
            major=major,
            classification=classification,
            previous_courses=previous_courses,
            current_gpa=gpa,
            credits_completed=max(0, credits),
            time_preference=np.random.choice(
                ["morning", "afternoon", "evening"]),
            preferred_days=selected_days,
            current_search_text=search_term,
            active_filters=selected_filters,
            session_hovers=[],
            session_watchlisted=[],
            session_paginations=0
        ))

    return students


def create_bandit_variants() -> Dict[str, CourseRecommendationBandit]:
    """Create different bandit configurations for A/B testing"""
    variants = {}

    variants['standard'] = CourseRecommendationBandit(
        MOCK_COURSES, learning_rate=0.01, epsilon=0.15, epsilon_decay=0.995
    )

    variants['high_exploration'] = CourseRecommendationBandit(
        MOCK_COURSES, learning_rate=0.01, epsilon=0.3, epsilon_decay=0.999
    )

    variants['fast_learning'] = CourseRecommendationBandit(
        MOCK_COURSES, learning_rate=0.05, epsilon=0.15, epsilon_decay=0.99
    )

    variants['conservative'] = CourseRecommendationBandit(
        MOCK_COURSES, learning_rate=0.005, epsilon=0.1, epsilon_decay=0.998
    )

    return variants


def simulate_learning_process(bandit: CourseRecommendationBandit,
                              generator: InteractionGenerator,
                              n_sessions: int = 1000) -> Tuple[List[UserInteraction], List[float]]:
    """Simulate the bandit learning process and track performance"""
    interactions = []
    reward_history = []

    print(f"Simulating {n_sessions} learning sessions...")

    for session_num, interaction in enumerate(generator.generate_interactions(n_sessions)):
        interactions.append(interaction)

        if interaction.course_idx != -1:
            bandit.update(
                interaction.student_context,
                [interaction.course_idx],
                [interaction.reward]
            )
            reward_history.append(interaction.reward)

        if (session_num + 1) % 200 == 0:
            avg_reward = np.mean(
                reward_history[-100:]) if len(reward_history) >= 100 else 0
            print(f"  Session {session_num +
                  1}: Recent avg reward = {avg_reward:.3f}")

    return interactions, reward_history


def run_ab_testing(variants: Dict[str, CourseRecommendationBandit],
                   students: List[StudentContext]) -> Dict[str, EvaluationMetrics]:
    """Run A/B testing between different bandit variants"""
    print("\n" + "="*60)
    print("RUNNING A/B TESTING")
    print("="*60)

    results = {}

    for variant_name, bandit in variants.items():
        print(f"\nTesting variant: {variant_name}")

        generator = InteractionGenerator(bandit, seed=42)

        interactions, _ = simulate_learning_process(
            bandit, generator, n_sessions=500)

        evaluator = OnlineEvaluator(k=5)
        metrics = evaluator.evaluate(bandit, interactions)
        results[variant_name] = metrics

        print(f"  Average reward: {metrics.average_reward:.4f}")
        print(f"  Click-through rate: {metrics.click_through_rate:.4f}")
        print(f"  Exploration rate: {metrics.exploration_rate:.4f}")

    return results


def run_comprehensive_evaluation(bandit: CourseRecommendationBandit,
                                 interactions: List[UserInteraction],
                                 generator: InteractionGenerator) -> Dict[str, EvaluationMetrics]:
    """Run all evaluation methods"""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE EVALUATION")
    print("="*60)

    results = {}

    print("\n1. Online Evaluation (using simulated interactions)")
    online_eval = OnlineEvaluator(k=5)
    results['online'] = online_eval.evaluate(bandit, interactions)
    print(f"   Average reward: {results['online'].average_reward:.4f}")
    print(f"   Regret: {results['online'].regret:.4f}")

    print("\n2. Offline Evaluation (IPS method)")
    offline_eval = OfflineEvaluator(k=5)
    results['offline'] = offline_eval.evaluate(bandit, interactions)
    print(f"   Average reward: {results['offline'].average_reward:.4f}")

    print("\n3. Simulation Evaluation (fresh synthetic data)")
    sim_eval = SimulationEvaluator(generator, n_episodes=300, k=5)
    results['simulation'] = sim_eval.evaluate(bandit)
    print(f"   Average reward: {results['simulation'].average_reward:.4f}")
    print(f"   Coverage: {results['simulation'].coverage:.4f}")

    if len(interactions) > 100:
        print("\n4. Cross-Validation (time-based splits)")
        cv_eval = CrossValidationEvaluator(
            n_folds=3, evaluator_class=OnlineEvaluator)
        cv_results = cv_eval.evaluate(bandit, interactions)

        avg_metrics_dict = {}
        for key in cv_results['fold_0'].to_dict().keys():
            avg_metrics_dict[key] = np.mean([
                cv_results[f'fold_{i}'].to_dict()[key] for i in range(3)
            ])
        results['cross_validation'] = EvaluationMetrics(**avg_metrics_dict)
        print(f"   CV Average reward: {
              results['cross_validation'].average_reward:.4f}")
        print(f"   CV Std: {
              np.std([cv_results[f'fold_{i}'].average_reward for i in range(3)]):.4f}")

    return results


def create_visualizations(ab_results: Dict[str, EvaluationMetrics],
                          eval_results: Dict[str, EvaluationMetrics],
                          reward_history: List[float],
                          save_dir: str = "evaluation_results"):
    """Create comprehensive visualizations"""
    Path(save_dir).mkdir(exist_ok=True)

    plt.style.use('default')
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        'A/B Testing Results: Bandit Variants Comparison', fontsize=16)

    variants = list(ab_results.keys())
    metrics_to_plot = ['average_reward',
                       'click_through_rate', 'exploration_rate', 'coverage']
    metric_titles = ['Average Reward', 'Click-Through Rate',
                     'Exploration Rate', 'Coverage']

    for i, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        values = [ab_results[variant].to_dict()[metric]
                  for variant in variants]
        bars = ax.bar(variants, values, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel('Value')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/ab_testing_results.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Evaluation Methods Comparison', fontsize=16)

    eval_methods = list(eval_results.keys())
    eval_metrics = ['average_reward', 'click_through_rate', 'precision_at_k',
                    'coverage', 'diversity', 'exploration_rate']

    for i, metric in enumerate(eval_metrics):
        row, col = i // 3, i % 3
        ax = axes[row, col]

        values = [eval_results[method].to_dict()[metric]
                  for method in eval_methods]
        bars = ax.bar(eval_methods, values, alpha=0.8)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Value')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/evaluation_methods_comparison.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 6))

    window_size = 50
    if len(reward_history) > window_size:
        moving_avg = pd.Series(reward_history).rolling(
            window=window_size).mean()
        plt.plot(moving_avg, label=f'Moving Average (window={
                 window_size})', linewidth=2)

    plt.plot(reward_history, alpha=0.3, label='Raw Rewards')
    plt.axhline(y=np.mean(reward_history), color='red', linestyle='--',
                label=f'Overall Average ({np.mean(reward_history):.3f})')

    plt.title('Bandit Learning Curve')
    plt.xlabel('Interaction Number')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/learning_curve.png", dpi=300, bbox_inches='tight')
    plt.show()


def save_results(ab_results: Dict[str, EvaluationMetrics],
                 eval_results: Dict[str, EvaluationMetrics],
                 save_dir: str = "evaluation_results"):
    """Save results to JSON files"""
    Path(save_dir).mkdir(exist_ok=True)

    ab_data = {variant: metrics.to_dict()
               for variant, metrics in ab_results.items()}
    eval_data = {method: metrics.to_dict()
                 for method, metrics in eval_results.items()}

    with open(f"{save_dir}/ab_testing_results.json", 'w') as f:
        json.dump(ab_data, f, indent=2)

    with open(f"{save_dir}/evaluation_results.json", 'w') as f:
        json.dump(eval_data, f, indent=2)

    print(f"\nResults saved to {save_dir}/")


def generate_report(ab_results: Dict[str, EvaluationMetrics],
                    eval_results: Dict[str, EvaluationMetrics]) -> str:
    """Generate comprehensive evaluation report"""

    report = []
    report.append("COURSE RECOMMENDATION BANDIT EVALUATION REPORT")
    report.append("=" * 55)
    report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("A/B TESTING RESULTS")
    report.append("-" * 30)

    best_variant = max(ab_results.items(), key=lambda x: x[1].average_reward)
    report.append(f"Best performing variant: {best_variant[0]}")
    report.append(f"Best average reward: {best_variant[1].average_reward:.4f}")
    report.append("")

    for variant, metrics in ab_results.items():
        report.append(f"{variant.upper()}:")
        report.append(f"  Average Reward: {metrics.average_reward:.4f}")
        report.append(
            f"  Click-Through Rate: {metrics.click_through_rate:.4f}")
        report.append(f"  Exploration Rate: {metrics.exploration_rate:.4f}")
        report.append(f"  Coverage: {metrics.coverage:.4f}")
        report.append("")

    report.append("EVALUATION METHODS COMPARISON")
    report.append("-" * 35)

    for method, metrics in eval_results.items():
        report.append(f"{method.upper()}:")
        report.append(f"  Average Reward: {metrics.average_reward:.4f}")
        report.append(f"  Cumulative Reward: {metrics.cumulative_reward:.2f}")
        report.append(
            f"  Click-Through Rate: {metrics.click_through_rate:.4f}")
        report.append(f"  Precision@K: {metrics.precision_at_k:.4f}")
        report.append(f"  Coverage: {metrics.coverage:.4f}")
        report.append(f"  Diversity: {metrics.diversity:.4f}")
        report.append("")

    report.append("RECOMMENDATIONS")
    report.append("-" * 20)

    if best_variant[1].exploration_rate < 0.1:
        report.append("⚠️  Consider increasing exploration rate")

    if best_variant[1].coverage < 0.3:
        report.append("⚠️  Low course coverage - diversify recommendations")

    if best_variant[1].average_reward > 0.3:
        report.append("✅ Good performance - consider production deployment")

    report.append("")
    report.append("END OF REPORT")

    return "\n".join(report)


def main():
    """Main evaluation script"""
    print("COURSE RECOMMENDATION BANDIT EVALUATION")
    print("=" * 50)
    print("This script runs comprehensive evaluation of the bandit system")
    print("using simulation, offline methods, and A/B testing.\n")

    output_dir = "evaluation_results"
    Path(output_dir).mkdir(exist_ok=True)

    print("1. Setting up evaluation environment...")
    students = create_diverse_student_contexts(n_students=150)
    bandit_variants = create_bandit_variants()
    print(f"   Created {len(students)} diverse student profiles")
    print(f"   Created {len(bandit_variants)} bandit variants")

    ab_results = run_ab_testing(bandit_variants, students)

    best_variant_name = max(
        ab_results.items(), key=lambda x: x[1].average_reward)[0]
    best_bandit = bandit_variants[best_variant_name]
    print(f"\nBest variant: {best_variant_name}")

    print(f"\n4. Generating interactions for detailed evaluation...")
    generator = InteractionGenerator(best_bandit, seed=42)
    interactions, reward_history = simulate_learning_process(
        best_bandit, generator, n_sessions=800)
    print(f"   Generated {len(interactions)} total interactions")
    print(f"   Average reward: {np.mean(reward_history):.4f}")

    eval_results = run_comprehensive_evaluation(
        best_bandit, interactions, generator)

    print(f"\n6. Creating visualizations...")
    create_visualizations(ab_results, eval_results, reward_history, output_dir)

    print(f"7. Saving results...")
    save_results(ab_results, eval_results, output_dir)

    print(f"\n8. Generating evaluation report...")
    report = generate_report(ab_results, eval_results)

    with open(f"{output_dir}/evaluation_report.txt", 'w') as f:
        f.write(report)

    print("\n" + report)

    print(f"\n{'='*50}")
    print("EVALUATION COMPLETE")
    print(f"{'='*50}")
    print(f"Results saved to: {output_dir}/")
    print("Files created:")
    print("  - ab_testing_results.json")
    print("  - evaluation_results.json")
    print("  - ab_testing_results.png")
    print("  - evaluation_methods_comparison.png")
    print("  - learning_curve.png")
    print("  - evaluation_report.txt")

    best_reward = max(
        metrics.average_reward for metrics in ab_results.values())
    print(f"\nBest average reward achieved: {best_reward:.4f}")
    print(f"Total interactions simulated: {len(interactions)}")
    print(f"Evaluation methods used: {len(eval_results)}")


if __name__ == "__main__":

    np.random.seed(42)

    import sys
    from types import ModuleType

    subjects_module = ModuleType('subjects')
    subjects_module.subjects = [
        'CS', 'MATH', 'PHYS', 'CHEM', 'BIOL', 'ENGL', 'HIST', 'PSYC',
        'ENGR', 'BUSI', 'ECON', 'PHIL', 'ARTS', 'MUSC', 'THEA'
    ] * 10
    sys.modules['subjects'] = subjects_module

    majors_module = ModuleType('majors')
    used_majors = [
        "Computer Science (BS)", "Mathematics (BS)", "Physics (BS)",
        "Engineering (BS)", "Biology (BS)", "Chemistry (BS)",
        "English (BA)", "Psychology (BS)", "Business (BS)"
    ]
    additional_majors = [
        "Economics (BS)", "Philosophy (BA)", "History (BA)", "Art (BA)", "Music (BA)",
        "Theatre (BA)", "Mechanical Engineering (BS)", "Electrical Engineering (BS)",
        "Civil Engineering (BS)", "Chemical Engineering (BS)", "Biomedical Engineering (BS)",
        "Accounting (BS)", "Finance (BS)", "Marketing (BS)", "Management (BS)",
        "Political Science (BA)", "Sociology (BA)", "Anthropology (BA)",
        "Environmental Science (BS)", "Geology (BS)", "Astronomy (BS)",
        "Statistics (BS)", "Applied Mathematics (BS)", "Pure Mathematics (BS)",
        "Molecular Biology (BS)", "Microbiology (BS)", "Biochemistry (BS)",
        "Physical Chemistry (BS)", "Organic Chemistry (BS)", "Analytical Chemistry (BS)",
        "Creative Writing (BA)", "Linguistics (BA)", "Comparative Literature (BA)",
        "Clinical Psychology (BS)", "Social Psychology (BS)", "Cognitive Psychology (BS)",
        "International Business (BS)", "Human Resources (BS)", "Operations Management (BS)",
        "Public Administration (BA)", "Criminal Justice (BA)", "Social Work (BA)"
    ]
    majors_module.majors = used_majors + additional_majors
    sys.modules['majors'] = majors_module

    section_module = ModuleType('section')
    section_module.courses = MOCK_COURSES
    sys.modules['section'] = section_module

    main()
