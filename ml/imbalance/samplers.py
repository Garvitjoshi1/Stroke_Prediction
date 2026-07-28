import logging

from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    ADASYN
)

from imblearn.combine import (
    SMOTEENN,
    SMOTETomek
)

logger = logging.getLogger(__name__)


class SamplerFactory:
    """
    Factory returning all sampling techniques.
    """

    @staticmethod
    def get_samplers(random_state=42):

        logger.info("Loading imbalance samplers...")

        return {

            "baseline": None,

            "random_over":
                RandomOverSampler(
                    random_state=random_state
                ),

            "smote":
                SMOTE(
                    random_state=random_state
                ),

            "borderline_smote":
                BorderlineSMOTE(
                    random_state=random_state
                ),

            "adasyn":
                ADASYN(
                    random_state=random_state
                ),

            "smoteenn":
                SMOTEENN(
                    random_state=random_state
                ),

            "smotetomek":
                SMOTETomek(
                    random_state=random_state
                )
        }