import evaluate
import datasets
import numpy as np

from Utils import compute_metrics

class RecallAtK(evaluate.Metric):
    """Computes recall@k for a given k."""

    def __init__(self, val_df, k=100, filter_by_lang=True, **kwargs):
        super().__init__(**kwargs)

        self.val_df = val_df
        self.k = k
        self.filter_by_lang = filter_by_lang

    def _info(self):

        return evaluate.MetricInfo(
            description="metric for MNR",
            citation="No citation",
            homepage="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions_a": datasets.Sequence(datasets.Value("float32")),
                    "predictions_b": datasets.Sequence(datasets.Value("float32")),
                }
            ),
        )

    def _compute(self, predictions_a, predictions_b):
        label_ids = None
        eval_predictions = ((np.array(predictions_a), np.array(predictions_b)), label_ids)
        return compute_metrics(eval_predictions, self.val_df, self.k, self.filter_by_lang)