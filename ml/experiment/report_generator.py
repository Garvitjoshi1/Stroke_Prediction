from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd


class ReportGenerator:

    def __init__(self, experiment_dir: Path):

        self.experiment_dir = Path(experiment_dir)

    # ------------------------------------------------ #

    def generate(
        self,
        config: dict,
        metrics: dict,
        leaderboard: pd.DataFrame,
        cv_results: pd.DataFrame,
        threshold_results: pd.DataFrame,
        imbalance_results: pd.DataFrame,
    ):

        report_path = self.experiment_dir / "REPORT.md"

        with open(report_path, "w", encoding="utf-8") as f:

            f.write("# NeuroGuard Experiment Report\n\n")

            f.write(
                f"Generated : {datetime.now()}\n\n"
            )

            f.write("---\n\n")

            f.write("## Configuration\n\n")

            for key, value in config.items():

                f.write(f"- **{key}** : {value}\n")

            f.write("\n---\n\n")

            f.write("## Final Metrics\n\n")

            f.write("|Metric|Value|\n")
            f.write("|------|-----|\n")

            for k, v in metrics.items():

                f.write(f"|{k}|{v:.4f}|\n")

            f.write("\n---\n\n")

            f.write("## Model Leaderboard\n\n")

            f.write(
                leaderboard.to_markdown(
                    index=False
                )
            )

            f.write("\n\n---\n\n")

            f.write("## Cross Validation\n\n")

            f.write(
                cv_results.to_markdown(
                    index=False
                )
            )

            f.write("\n\n---\n\n")

            f.write("## Threshold Search\n\n")

            f.write(
                threshold_results.to_markdown(
                    index=False
                )
            )

            f.write("\n\n---\n\n")

            f.write("## Imbalance Experiments\n\n")

            f.write(
                imbalance_results.to_markdown(
                    index=False
                )
            )

            f.write("\n\n---\n\n")

            f.write("## Best Model\n\n")

            best_model = leaderboard.iloc[0]

            f.write(
                f"- Model : **{best_model['model']}**\n"
            )

            f.write(
                f"- ROC AUC : **{best_model['roc_auc']:.4f}**\n"
            )

            f.write(
                f"- Recall : **{best_model['recall']:.4f}**\n"
            )

            f.write(
                f"- Precision : **{best_model['precision']:.4f}**\n"
            )

            f.write("\n---\n\n")

            f.write("## Recommendation\n\n")

            f.write(
                "The selected model demonstrates the "
                "best trade-off between discrimination "
                "ability (ROC-AUC), recall, and "
                "generalization performance. "
                "Future work should focus on:\n\n"
            )

            f.write("- Hyperparameter Optimization\n")
            f.write("- Probability Calibration\n")
            f.write("- Explainability using SHAP\n")
            f.write("- External Validation Dataset\n")
            f.write("- API Deployment\n")

        return report_path