import pandas as pd
import traceback

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