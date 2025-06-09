import pandas as pd
import traceback
import pickle
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, Ridge
from catboost import CatBoostRegressor
from typing import Union, Literal
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class Netmodel_52(nn.Module):
    def __init__(self, input_dim):
        super(Netmodel_52, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def create_model(model_type: Literal["linear_model", "neural_network", "catboost_model"]) -> Union[LinearRegression, Netmodel_52, CatBoostRegressor]:   
    """
    Create a model instance based on the specified type and load its parameters.
    
    Args:
        model_type (str): Type of model to create. One of:
            - "linear_model": Basic LinearRegression model
            - "catboost_model": Gradient boosting model
            - "neural_network": Neural network model
    
    Returns:
        Union[LinearRegression, Netmodel_52, CatBoostRegressor]: Initialized model with loaded parameters
    
    Raises:
        ValueError: If model_type is invalid or model files are not found
    """
    try:
        if model_type in ["linear_model", "catboost_model"]:
            # Load sklearn model parameters from .pkl file
            model_path = f"resources/models/{model_type}.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
            
        elif model_type == "neural_network":
            # Load neural network checkpoint
            checkpoint_path = "resources/models/neural_network.pt"
            
            # Create model
            model = Netmodel_52(input_dim=4237)  # Adjust input_dim based on your model architecture
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to evaluation mode
            return model
            
        else:
            raise ValueError(f"Invalid model type: {model_type}. Must be one of: linear_model, catboost_model, neural_network")
            
    except FileNotFoundError as e:
        raise ValueError(f"Model files not found for {model_type}: {str(e)}")
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Error creating model {model_type}: {str(e)}")

def preprocess_data_linear_model(data: pd.DataFrame, ohe) -> pd.DataFrame:
    try:
        # Преобразуем типы данных
        data['settlement'] = data['settlement'].astype(str)
        data['lat'] = data['lat'].astype(float)
        data['long'] = data['long'].astype(float)
        data['settlement_count'] = data['settlement_count'].astype(int)
        data['atm_group'] = data['atm_group'].astype(str)
        data['postal_code'] = data['postal_code'].astype(str)

        # Категориальные признаки
        cat_features = ['settlement', 'street_name', 'atm_group', 'postal_code']

        # Преобразуем с помощью OHE
        encoded = ohe.transform(data[cat_features]).toarray()
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_features), index=data.index)

        # Объединяем обработанные данные
        return pd.concat([data.drop(columns=cat_features), encoded_df], axis=1)

    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Ошибка обработки данных: {str(e)}")
    
def preprocess_data_catboost_model(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data = data[(data['lat'].notnull()) & (data['long'].notnull())]
        data['settlement_count'].fillna(round(data['settlement_count'].median(), 2), inplace=True)
        data['atm_group'] = data['atm_group'].astype(object)
        data['postal_code'] = data['postal_code'].astype(object)
        data['number'] = data['number'].astype(object)
        data['len_postal_code'] = data['len_postal_code'].astype(object)
        data['address_rus'] = data['address_rus'].astype(object)
        data['address'] = data['address'].astype(object)
        data['id'] = data['id'].astype(object)
        data['country'] = data['country'].astype(object)
        X = data.drop(columns=['id', 'address', 'address_rus', 'len_postal_code', 'country', 'target', 'feature_mean']).copy()
        X['p'].fillna(X['p'].median(), inplace=True)
        X['street_name'].fillna('999', inplace=True)
        X['settlement'].fillna('111', inplace=True)
        X['postal_code'].fillna('-999', inplace=True)
        X['number'].fillna('222', inplace=True)
        X['number'] = X['number'].astype(str)
        X['number'] = X['number'].apply(lambda row: row.mode() if pd.isna(row) else row)
        X.drop('len_address', axis=1, inplace=True)
        X['atm_group'] = X['atm_group'].astype(float).astype(int)
        X['atm_group'] = X['atm_group'].astype(str)
        return X

    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Ошибка обработки данных: {str(e)}")

def preprocess_data_neural_network(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data = data[(data['lat'].notnull()) & (data['long'].notnull())]
        data['settlement_count'].fillna(round(data['settlement_count'].median(), 2), inplace=True)
        data['atm_group'] = data['atm_group'].astype(object)
        data['postal_code'] = data['postal_code'].astype(object)
        data['number'] = data['number'].astype(object)
        data['len_postal_code'] = data['len_postal_code'].astype(object)
        data['address_rus'] = data['address_rus'].astype(object)
        data['address'] = data['address'].astype(object)
        data['id'] = data['id'].astype(object)
        data['country'] = data['country'].astype(object)
        data['p'].fillna(data['p'].median(), inplace=True)
        data['street_name'].fillna('999', inplace=True)
        data['settlement'].fillna('222', inplace=True)
        data['postal_code'].fillna('-999', inplace=True)
        data['number'].fillna('333', inplace=True)
        
        cat_features = ['atm_group', 'settlement', 'street_name', 'postal_code', 'number']
        num_features = ['lat', 'long', 'settlement_count', 'p'] 
        features = cat_features + num_features
        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), num_features),
                                               ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)])
        
        X = data[features]
        X_train_scaled = preprocessor.fit_transform(X)
        return X_train_scaled

    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Ошибка обработки данных: {str(e)}")