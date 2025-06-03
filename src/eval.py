import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import ndcg_score
import seaborn as sns


from model import (
    StudentContext, CourseRecommendationBandit,
    ClassificationLevel, CourseAction
)
from interaction import InteractionGenerator, UserInteraction, InteractionType


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    cumulative_reward: float
    average_reward: float
    regret: float
    click_through_rate: float
    watchlist_rate: float
    coverage: float
    diversity: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    exploration_rate: float

    def to_dict(self) -> Dict:
        return {
            'cumulative_reward': self.cumulative_reward,
            'average_reward': self.average_reward,
            'regret': self.regret,
            'click_through_rate': self.click_through_rate,
            'watchlist_rate': self.watchlist_rate,
            'coverage': self.coverage,
            'diversity': self.diversity,
            'precision_at_k': self.precision_at_k,
            'recall_at_k': self.recall_at_k,
            'ndcg_at_k': self.ndcg_at_k,
            'exploration_rate': self.exploration_rate
        }


class BaseEvaluator(ABC):
    """Base class for bandit evaluation methods"""

    @abstractmethod
    def evaluate(self, bandit: CourseRecommendationBandit,
                 interactions: List[UserInteraction]) -> EvaluationMetrics:
        pass


class OnlineEvaluator(BaseEvaluator):
    """
    Online evaluation using actual user interactions.
    This is the most realistic but requires live data.
    """

    def __init__(self, k: int = 5):
        self.k = k

    def evaluate(self, bandit: CourseRecommendationBandit,
                 interactions: List[UserInteraction]) -> EvaluationMetrics:
        """Evaluate bandit using online interactions"""

        total_reward = 0.0
        total_recommendations = 0
        clicks = 0
        watchlists = 0
        recommended_courses = set()
        subjects_recommended = set()
        positive_interactions = 0
        total_relevant = 0
        exploration_count = 0

        oracle_rewards = []
        actual_rewards = []

        for interaction in interactions:
            if interaction.course_idx != -1:
                total_reward += interaction.reward
                actual_rewards.append(interaction.reward)

                oracle_reward = self._calculate_oracle_reward(
                    interaction.student_context,
                    interaction.recommended_courses,
                    bandit
                )
                oracle_rewards.append(oracle_reward)

                if interaction.interaction_type == InteractionType.CLICK:
                    clicks += 1
                    positive_interactions += 1
                elif interaction.interaction_type == InteractionType.WATCHLIST:
                    watchlists += 1
                    positive_interactions += 1

                recommended_courses.add(interaction.course_idx)
                course = bandit.course_actions[interaction.course_idx]
                subjects_recommended.add(course.course_subject)

                predicted_rewards = bandit.predict_rewards(
                    interaction.student_context)
                best_courses = np.argsort(
                    predicted_rewards)[-len(interaction.recommended_courses):]
                if interaction.course_idx not in best_courses:
                    exploration_count += 1

            total_recommendations += len(interaction.recommended_courses)

        n_interactions = len([i for i in interactions if i.course_idx != -1])

        metrics = EvaluationMetrics(
            cumulative_reward=total_reward,
            average_reward=total_reward / max(n_interactions, 1),
            regret=sum(oracle_rewards) -
            sum(actual_rewards) if oracle_rewards else 0.0,
            click_through_rate=clicks / max(n_interactions, 1),
            watchlist_rate=watchlists / max(n_interactions, 1),
            coverage=len(recommended_courses) / len(bandit.course_actions),
            diversity=len(subjects_recommended) /
            len(bandit._get_all_subjects()),
            precision_at_k=positive_interactions / max(n_interactions, 1),
            recall_at_k=positive_interactions /
            max(total_relevant or n_interactions, 1),
            ndcg_at_k=self._calculate_ndcg(interactions, bandit),
            exploration_rate=exploration_count / max(n_interactions, 1)
        )

        return metrics

    def _calculate_oracle_reward(self, context: StudentContext,
                                 recommended_courses: List[int],
                                 bandit: CourseRecommendationBandit) -> float:
        """Calculate the best possible reward for this context"""
        all_rewards = bandit.predict_rewards(context)
        return np.max(all_rewards[recommended_courses])

    def _calculate_ndcg(self, interactions: List[UserInteraction],
                        bandit: CourseRecommendationBandit) -> float:
        """Calculate NDCG@k for the recommendations"""
        if not interactions:
            return 0.0

        ndcg_scores = []
        for interaction in interactions:
            if interaction.course_idx != -1:

                relevance = []
                for course_idx in interaction.recommended_courses:
                    if course_idx == interaction.course_idx:
                        if interaction.interaction_type in [InteractionType.CLICK, InteractionType.WATCHLIST]:
                            relevance.append(2)
                        elif interaction.interaction_type == InteractionType.HOVER:
                            relevance.append(1)
                        else:
                            relevance.append(0)
                    else:
                        relevance.append(0)

                if sum(relevance) > 0:

                    true_relevance = np.array([relevance])
                    predicted_scores = bandit.predict_rewards(
                        interaction.student_context)
                    pred_scores = np.array(
                        [predicted_scores[interaction.recommended_courses]])

                    try:
                        ndcg = ndcg_score(
                            true_relevance, pred_scores, k=self.k)
                        ndcg_scores.append(ndcg)
                    except:
                        pass

        return np.mean(ndcg_scores) if ndcg_scores else 0.0


class OfflineEvaluator(BaseEvaluator):
    """
    Offline evaluation using historical interaction data.
    Uses techniques like Inverse Propensity Scoring (IPS).
    """

    def __init__(self, logging_policy: Optional[Callable] = None, k: int = 5):

        self.logging_policy = logging_policy
        self.k = k

    def evaluate(self, bandit: CourseRecommendationBandit,
                 interactions: List[UserInteraction]) -> EvaluationMetrics:
        """Offline evaluation using IPS and direct method"""

        dm_rewards = []

        ips_rewards = []

        total_interactions = 0
        positive_interactions = 0

        for interaction in interactions:
            if interaction.course_idx != -1:
                total_interactions += 1

                predicted_reward = bandit.predict_rewards(interaction.student_context)[
                    interaction.course_idx]
                dm_rewards.append(predicted_reward)

                propensity = 1.0 / len(interaction.recommended_courses)
                ips_reward = interaction.reward / propensity
                ips_rewards.append(ips_reward)

                if interaction.interaction_type in [InteractionType.CLICK, InteractionType.WATCHLIST]:
                    positive_interactions += 1

        dr_rewards = []
        for dm_reward, ips_reward in zip(dm_rewards, ips_rewards):
            dr_reward = dm_reward + ips_reward - dm_reward
            dr_rewards.append(dr_reward)

        avg_reward = np.mean(dr_rewards) if dr_rewards else 0.0

        metrics = EvaluationMetrics(
            cumulative_reward=sum(dr_rewards),
            average_reward=avg_reward,
            regret=0.0,
            click_through_rate=positive_interactions /
            max(total_interactions, 1),
            watchlist_rate=0.0,
            coverage=0.0,
            diversity=0.0,
            precision_at_k=positive_interactions / max(total_interactions, 1),
            recall_at_k=0.0,
            ndcg_at_k=0.0,
            exploration_rate=0.0
        )

        return metrics


class SimulationEvaluator(BaseEvaluator):
    """
    Simulation-based evaluation using synthetic environments.
    Good for controlled experiments and A/B testing.
    """

    def __init__(self, simulator: InteractionGenerator, n_episodes: int = 1000, k: int = 5):
        self.simulator = simulator
        self.n_episodes = n_episodes
        self.k = k

    def evaluate(self, bandit: CourseRecommendationBandit,
                 interactions: List[UserInteraction] = None) -> EvaluationMetrics:
        """Evaluate using simulation"""

        eval_interactions = list(
            self.simulator.generate_interactions(self.n_episodes))

        online_eval = OnlineEvaluator(self.k)
        return online_eval.evaluate(bandit, eval_interactions)


class CrossValidationEvaluator:
    """
    Time-based cross-validation for temporal data.
    Splits interactions by time periods.
    """

    def __init__(self, n_folds: int = 5, evaluator_class: BaseEvaluator = OnlineEvaluator):
        self.n_folds = n_folds
        self.evaluator_class = evaluator_class

    def evaluate(self, bandit: CourseRecommendationBandit,
                 interactions: List[UserInteraction]) -> Dict[str, EvaluationMetrics]:
        """Perform time-based cross-validation"""

        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)

        fold_size = len(sorted_interactions) // self.n_folds
        results = {}

        for fold in range(self.n_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < self.n_folds - \
                1 else len(sorted_interactions)

            train_interactions = (sorted_interactions[:start_idx] +
                                  sorted_interactions[end_idx:])
            test_interactions = sorted_interactions[start_idx:end_idx]

            fold_bandit = self._train_bandit_copy(bandit, train_interactions)

            evaluator = self.evaluator_class()
            metrics = evaluator.evaluate(fold_bandit, test_interactions)
            results[f'fold_{fold}'] = metrics

        return results

    def _train_bandit_copy(self, original_bandit: CourseRecommendationBandit,
                           train_interactions: List[UserInteraction]) -> CourseRecommendationBandit:
        """Create and train a copy of the bandit"""

        new_bandit = CourseRecommendationBandit(
            original_bandit.available_courses,
            original_bandit.learning_rate,
            original_bandit.epsilon,
            original_bandit.epsilon_decay
        )

        for interaction in train_interactions:
            if interaction.course_idx != -1:
                new_bandit.update(
                    interaction.student_context,
                    [interaction.course_idx],
                    [interaction.reward]
                )

        return new_bandit


class ComprehensiveEvaluationSuite:
    """
    Complete evaluation suite that runs multiple evaluation methods
    and provides comprehensive analysis.
    """

    def __init__(self):
        self.results = {}

    def run_evaluation(self, bandit: CourseRecommendationBandit,
                       interactions: List[UserInteraction],
                       simulator: Optional[InteractionGenerator] = None) -> Dict[str, EvaluationMetrics]:
        """Run comprehensive evaluation"""

        results = {}

        print("Running online evaluation...")
        online_eval = OnlineEvaluator()
        results['online'] = online_eval.evaluate(bandit, interactions)

        print("Running offline evaluation...")
        offline_eval = OfflineEvaluator()
        results['offline'] = offline_eval.evaluate(bandit, interactions)

        if simulator:
            print("Running simulation evaluation...")
            sim_eval = SimulationEvaluator(simulator, n_episodes=500)
            results['simulation'] = sim_eval.evaluate(bandit)

        if len(interactions) > 100:
            print("Running cross-validation...")
            cv_eval = CrossValidationEvaluator(n_folds=3)
            cv_results = cv_eval.evaluate(bandit, interactions)

            avg_metrics = self._average_cv_results(cv_results)
            results['cross_validation'] = avg_metrics

        self.results = results
        return results

    def _average_cv_results(self, cv_results: Dict[str, EvaluationMetrics]) -> EvaluationMetrics:
        """Average cross-validation results"""
        all_metrics = [metrics.to_dict() for metrics in cv_results.values()]
        avg_dict = {}

        for key in all_metrics[0].keys():
            avg_dict[key] = np.mean([m[key] for m in all_metrics])

        return EvaluationMetrics(**avg_dict)

    def plot_results(self, save_path: Optional[str] = None):
        """Create comprehensive plots of evaluation results"""
        if not self.results:
            print("No results to plot. Run evaluation first.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Bandit Evaluation Results', fontsize=16)

        methods = list(self.results.keys())
        metrics_data = {}

        key_metrics = ['average_reward', 'click_through_rate', 'coverage',
                       'diversity', 'precision_at_k', 'exploration_rate']

        for metric in key_metrics:
            metrics_data[metric] = [self.results[method].to_dict()[metric]
                                    for method in methods]

        for i, metric in enumerate(key_metrics):
            row, col = i // 3, i % 3
            ax = axes[row, col]

            bars = ax.bar(methods, metrics_data[metric])
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Value')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            for bar, value in zip(bars, metrics_data[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self) -> str:
        """Generate a text report of evaluation results"""
        if not self.results:
            return "No evaluation results available."

        report = "BANDIT EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"

        for method, metrics in self.results.items():
            report += f"{method.upper()} EVALUATION:\n"
            report += "-" * 30 + "\n"

            metrics_dict = metrics.to_dict()
            for metric, value in metrics_dict.items():
                report += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"

            report += "\n"

        report += "INTERPRETATION:\n"
        report += "-" * 30 + "\n"

        best_method = max(self.results.items(),
                          key=lambda x: x[1].average_reward)
        report += f"Best performing method: {best_method[0]} "
        report += f"(avg reward: {best_method[1].average_reward:.4f})\n\n"

        if best_method[1].exploration_rate < 0.1:
            report += "⚠️  Low exploration rate detected - model may be under-exploring\n"

        if best_method[1].coverage < 0.3:
            report += "⚠️  Low coverage detected - model may be too focused on few courses\n"

        return report


def demo_evaluation():
    """Demonstrate the evaluation framework"""

    print("Bandit Evaluation Framework Demo")
    print("=" * 40)

    print("\nEvaluation Methods Available:")
    print("1. Online Evaluation - Uses real user interactions")
    print("2. Offline Evaluation - Uses historical data with IPS")
    print("3. Simulation Evaluation - Uses synthetic interactions")
    print("4. Cross-Validation - Time-based splits")
    print("5. Comprehensive Suite - Runs all methods")

    print("\nKey Metrics Computed:")
    metrics = [
        "Cumulative Reward", "Average Reward", "Regret",
        "Click-Through Rate", "Watchlist Rate", "Coverage",
        "Diversity", "Precision@K", "Recall@K", "NDCG@K",
        "Exploration Rate"
    ]

    for metric in metrics:
        print(f"  • {metric}")

    print("\nExample Usage:")
    print("""
    
    eval_suite = ComprehensiveEvaluationSuite()
    
    
    results = eval_suite.run_evaluation(bandit, interactions, simulator)
    
    
    eval_suite.plot_results('evaluation_results.png')
    report = eval_suite.generate_report()
    print(report)
    """)


if __name__ == "__main__":
    demo_evaluation()
