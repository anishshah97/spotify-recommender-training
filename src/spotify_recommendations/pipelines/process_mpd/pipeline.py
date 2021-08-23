from kedro.pipeline import Pipeline, node

from .clean_mpd import multiprocess_clean_mpd


def prepare_mpd_dataset(**kwargs):
    return Pipeline([
        node(
            multiprocess_clean_mpd,
            inputs=[],
            outputs="completed_result_notifications"
        ),

    ])
