import pandas as pd
import traceback
import pickle
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, Ridge
from typing import Union, Literal

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

def create_model(model_type: Literal["linear_model", "neural_network"]) -> Union[LinearRegression, Netmodel_52]:   
    """
    Create a model instance based on the specified type and load its parameters.
    
    Args:
        model_type (str): Type of model to create. One of:
            - "linear_model": Basic LinearRegression model
            - "linear_model_scaled": LinearRegression model with scaled features
            - "ridge": Ridge regression model
            - "neural_network": Neural network model
    
    Returns:
        Union[LinearRegression, Ridge, ATMNeuralNetwork]: Initialized model with loaded parameters
    
    Raises:
        ValueError: If model_type is invalid or model files are not found
    """
    try:
        if model_type in ["linear_model", "ridge"]:
            # Load sklearn model parameters from .pkl file
            model_path = f"resources/models/{model_type}.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
            
        elif model_type == "neural_network":
            # Load neural network state dict
            state_dict_path = "resources/models/neural_network.pth"
            
            # Create and load model
            model = Netmodel_52(input_dim=8)
            model.load_state_dict(torch.load(state_dict_path))
            model.eval()  # Set to evaluation mode
            return model
            
        else:
            raise ValueError(f"Invalid model type: {model_type}. Must be one of: linear_model, linear_model_scaled, ridge, neural_network")
            
    except FileNotFoundError as e:
        raise ValueError(f"Model files not found for {model_type}: {str(e)}")
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Error creating model {model_type}: {str(e)}")

def preprocess_data(data: pd.DataFrame, ohe) -> pd.DataFrame:
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