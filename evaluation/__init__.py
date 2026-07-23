from .metrics import SegmentationEvaluator, format_results
from .engineering import profile_model, EngineeringReport
from .tusimple_metrics import TuSimpleEvaluator, format_results as format_tusimple_results

# Maps a dataset name (data.datasets.DATASETS key) to its own official
# evaluator + result formatter. scripts/evaluate.py resolves through this
# instead of importing any one dataset's metric module directly, so
# adding a new dataset's metric doesn't require touching evaluate.py.
EVALUATORS = {
    "tusimple": (TuSimpleEvaluator, format_tusimple_results),
}


def build_evaluator(dataset_name: str):
    """Returns (evaluator_instance, format_results_fn) for a dataset's official metric."""
    if dataset_name not in EVALUATORS:
        raise ValueError(
            f"No official evaluator registered for dataset '{dataset_name}'. "
            f"Available: {list(EVALUATORS)}"
        )
    evaluator_cls, format_fn = EVALUATORS[dataset_name]
    return evaluator_cls(), format_fn
