import numpy as np
import pandas as pd
from typing import Tuple, Union, List

from helpers import helpers

import xgboost as xgb


class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
    
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
             
        if target_column:
            
            data['period_day'] = data['Fecha-I'].apply( helpers.get_period_day)
            data['high_season'] = data['Fecha-I'].apply(helpers.is_high_season)
            data['min_diff'] = data.apply(helpers.get_min_diff, axis = 1)
            
            threshold_in_minutes = 15
            data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
            
                    
            features = pd.concat([
                pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
                pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
                pd.get_dummies(data['MES'], prefix = 'MES')], 
                axis = 1
            )
            
            target = data[target_column]
                                    
            return features, target
                        
        else:
            return features
        
        
    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0/n_y1
        
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
                
        self._model.fit(features[helpers.FEATURES_COLS], target)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        
        xgboost_y_preds = self._model.predict(features[helpers.FEATURES_COLS])
        
        list_of_ints: List[int] = xgboost_y_preds.tolist()
        
        return list_of_ints

