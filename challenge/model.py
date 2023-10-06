import numpy as np
import pandas as pd
from typing import Tuple, Union, List
from datetime import datetime
import xgboost as xgb


class DelayModel:

    def __init__(
        self
    ):
        self._model = xgb.XGBClassifier # Model should be saved in this attribute.
        
        
        
            
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
        FEATURES_COLS = [
                "OPERA_Latin American Wings", 
                "MES_7",
                "MES_10",
                "OPERA_Grupo LATAM",
                "MES_12",
                "TIPOVUELO_I",
                "MES_4",
                "MES_11",
                "OPERA_Sky Airline",
                "OPERA_Copa Air"
            ]
        
        def get_period_day(date):
            date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
            morning_min = datetime.strptime("05:00", '%H:%M').time()
            morning_max = datetime.strptime("11:59", '%H:%M').time()
            afternoon_min = datetime.strptime("12:00", '%H:%M').time()
            afternoon_max = datetime.strptime("18:59", '%H:%M').time()
            evening_min = datetime.strptime("19:00", '%H:%M').time()
            evening_max = datetime.strptime("23:59", '%H:%M').time()
            night_min = datetime.strptime("00:00", '%H:%M').time()
            night_max = datetime.strptime("4:59", '%H:%M').time()
            
            if(date_time > morning_min and date_time < morning_max):
                return 'mañana'
            elif(date_time > afternoon_min and date_time < afternoon_max):
                return 'tarde'
            elif(
                (date_time > evening_min and date_time < evening_max) or
                (date_time > night_min and date_time < night_max)
            ):
                return 'noche'
         
        def is_high_season(fecha):
            fecha_año = int(fecha.split('-')[0])
            fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
            range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
            range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
            range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
            range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
            range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
            range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
            range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
            range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
            
            if ((fecha >= range1_min and fecha <= range1_max) or 
                (fecha >= range2_min and fecha <= range2_max) or 
                (fecha >= range3_min and fecha <= range3_max) or
                (fecha >= range4_min and fecha <= range4_max)):
                return 1
            else:
                return 0
            
        
        def get_min_diff(data: pd.DataFrame):
            fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
            fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
            min_diff = ((fecha_o - fecha_i).total_seconds())/60
            return min_diff
        
       
        def get_rate_from_column(data: pd.DataFrame, column):
            delays = {}
            for _, row in data.iterrows():
                if row['delay'] == 1:
                    if row[column] not in delays:
                        delays[row[column]] = 1
                    else:
                        delays[row[column]] += 1
            total = data[column].value_counts().to_dict()
            
            rates = {}
            for name, total in total.items():
                if name in delays:
                    rates[name] = round(total / delays[name], 2)
                else:
                    rates[name] = 0
                    
            return pd.DataFrame.from_dict(data = rates, orient = 'index', columns = ['Tasa (%)'])

        data['period_day'] = data['Fecha-I'].apply(get_period_day)
        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        
                
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )
        
                
        features = features[FEATURES_COLS]  
             
        if target_column != None:   
                      
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
                
        self._model.fit(features, target)

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
               
        
        xgboost_y_preds = self._model.predict(features)
        
        list_of_ints: List[int] = xgboost_y_preds.tolist()
        
        return list_of_ints

