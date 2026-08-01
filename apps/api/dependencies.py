from functools import lru_cache
import logging

from ml.inference import StrokePredictor

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_predictor() -> StrokePredictor:
    
    logger.info("=" * 60)
    logger.info("Initializing Stroke Predictor...")
    logger.info("=" * 60)

    predictor = StrokePredictor()

    logger.info("Stroke Predictor Ready.")

    return predictor