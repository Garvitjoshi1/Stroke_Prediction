from pathlib import Path

class ArtifactManager:

    def __init__(self, experiment_dir: Path):

        self.root = Path(experiment_dir)

        self.models = self.root / "models"

        self.data = self.root / "data"

        self.plots = self.root / "plots"

        self.logs = self.root / "logs"

        self.config = self.root / "config"

        self._create_directories()

    def _create_directories(self):

        for directory in [

            self.models,

            self.data,

            self.plots,

            self.logs,

            self.config

        ]:

            directory.mkdir(
                parents=True,
                exist_ok=True
            )

    def model(self, name: str):

        return self.models / f"{name}.joblib"

    def dataframe(self, name: str):

        return self.data / f"{name}.csv"

    def metrics(self):

        return self.data / "metrics.json"

    def leaderboard(self):

        return self.data / "leaderboard.csv"

    def threshold(self):

        return self.data / "threshold_search.csv"

    def imbalance(self):

        return self.data / "imbalance_results.csv"

    def cross_validation(self):

        return self.data / "cross_validation.csv"

    def roc_curve(self):

        return self.plots / "roc_curve.png"

    def pr_curve(self):

        return self.plots / "precision_recall_curve.png"

    def confusion_matrix(self):

        return self.plots / "confusion_matrix.png"

    def shap_summary(self):

        return self.plots / "shap_summary.png"

    def shap_waterfall(self):

        return self.plots / "shap_waterfall.png"

    def feature_importance(self):

        return self.plots / "feature_importance.png"

    def calibration_curve(self):

        return self.plots / "calibration_curve.png"

    def log_file(self):

        return self.logs / "experiment.log"

    def config_file(self):

        return self.config / "config.json"

    def report(self):

        return self.root / "REPORT.md"