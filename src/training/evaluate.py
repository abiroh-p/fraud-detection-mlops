"""
Model evaluation for the fraud detection pipeline.

Computes metrics that matter for fraud detection specifically.
Standard accuracy is useless here — a model predicting 'not fraud'
for everything achieves 95% accuracy while catching zero fraud.

Metrics we care about:
- Precision: Of all transactions flagged as fraud, how many were real?
- Recall: Of all actual fraud, how many did we catch?
- F1: Harmonic mean of precision and recall
- AUCPR: Area under Precision-Recall curve — best for imbalanced data
- AUC-ROC: Standard ranking metric
- Average Precision: Summarizes the PR curve in one number
"""

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """
    All evaluation metrics for one model run.

    Stored as a dataclass so we can call asdict() and log
    the entire thing to MLflow in one line.
    """

    precision: float
    recall: float
    f1: float
    roc_auc: float
    avg_precision: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    @property
    def fraud_catch_rate(self) -> float:
        """
        Percentage of actual fraud cases caught.
        This is recall — the most business-critical metric.
        Missing fraud is more costly than false alarms.
        """
        total_fraud = self.true_positives + self.false_negatives
        if total_fraud == 0:
            return 0.0
        return self.true_positives / total_fraud

    def log_summary(self) -> None:
        """Print a human-readable evaluation summary."""
        logger.info("── Evaluation Metrics ───────────────────────")
        logger.info("  Precision       : %.4f", self.precision)
        logger.info("  Recall          : %.4f", self.recall)
        logger.info("  F1 Score        : %.4f", self.f1)
        logger.info("  ROC-AUC         : %.4f", self.roc_auc)
        logger.info("  Avg Precision   : %.4f", self.avg_precision)
        logger.info("  Fraud Catch Rate: %.2f%%", self.fraud_catch_rate * 100)
        logger.info(
            "  Confusion Matrix: TP=%d TN=%d FP=%d FN=%d",
            self.true_positives,
            self.true_negatives,
            self.false_positives,
            self.false_negatives,
        )

    def to_dict(self) -> dict[str, float]:
        """
        Convert to flat dict for MLflow logging.
        MLflow expects metric values as plain floats.
        """
        d = asdict(self)
        # Add the property — asdict() only captures dataclass fields
        d["fraud_catch_rate"] = self.fraud_catch_rate
        return {k: float(v) for k, v in d.items()}


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """
    Evaluate a trained model on the test set.

    We use two outputs from the model:
    - predict_proba() → probability scores → used for AUC metrics
    - predict()       → binary labels     → used for confusion matrix

    The threshold parameter controls the decision boundary.
    In fraud detection, lowering the threshold (e.g. 0.3) catches
    more fraud but increases false positives — a business decision,
    not a technical one.

    Args:
        model: Trained sklearn-compatible model with predict_proba().
        X_test: Feature matrix for the test set.
        y_test: True labels for the test set.
        threshold: Probability cutoff for classifying as fraud.

    Returns:
        EvaluationMetrics instance with all computed metrics.
    """
    # Probability of fraud (class 1) for each transaction
    y_proba = model.predict_proba(X_test)[:, 1]

    # Binary predictions using the threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix gives us TP, TN, FP, FN in one call
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = EvaluationMetrics(
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_proba),
        avg_precision=average_precision_score(y_test, y_proba),
        true_positives=int(tp),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
    )

    metrics.log_summary()
    return metrics
