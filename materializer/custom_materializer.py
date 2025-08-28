import os
import pickle
from typing import Any, Type, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from zenml.materializers.base_materializer import BaseMaterializer   
from zenml.io import fileio

DEFAULT_FILENAME = "CustomerSatisfactionEnvironment.pkl"


class CustomerSatisfactionMaterializer(BaseMaterializer):
    """
    Custom materializer for the Customer Satisfaction Project.
    Handles saving and loading models, arrays, and dataframes.
    """

    ASSOCIATED_TYPES = (
        str,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        CatBoostRegressor,
        RandomForestRegressor,
        LGBMRegressor,
        XGBRegressor,
    )

    def handle_input(self, data_type: Type[Any]) -> Any:
        """Reads the object from the artifact store and returns it."""
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            obj = pickle.load(fid)
        return obj

    def handle_return(self, obj: Any) -> None:
        """Saves the object to the artifact store."""
        super().handle_return(obj)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(obj, fid)
