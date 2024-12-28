import streamlit as st
import pandas as pd
import requests
import io
import time
import streamlit.components.v1 as components
# URL FastAPI, который будет принимать файл и возвращать результат
backend_url = "http://127.0.0.1:8002/predict/"
# Ожидаемые столбцы в CSV
expected_columns = {
    "lat": float,
    "long": float,
    "settlement": str,
    "street_name": str,
    "settlement_count": int,
    "atm_group": int,
    "postal_code": int}



def validate_csv(file):
    try:
        # Читаем CSV файл
        df = pd.read_csv(file, encoding='windows-1251')
        # Проверяем, есть ли в файле все ожидаемые колонки
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            return None, False, f"Отсутствуют необходимые колонки: {', '.join(missing_columns)}"
        return df, True, "CSV файл валиден"
    except Exception as e:
        return None, False, f"Ошибка при обработке файла: {str(e)}"


def display_data():
    """Функция для отображения данных и их описательной статистики."""
    df = st.session_state["dataframe"]
    st.header("Ваши данные:")
    st.write(f"- Загруженных объектов: {df.shape[0]}")
    st.write(df.head(5))  # Показываем первые строки данных
    st.write(f"- Дублей в данных: {df.duplicated().sum()}")
    st.write("- Описательная статистика:")
    st.write(df.describe(include='all'))


with open("heatmap.html", "r", encoding="utf-8") as f:
    map_html = f.read()

def main():
    st.title("Определение популярности геолокации для размещения банкомата")
    st.markdown("""
    Данная работа включает в себя изучение чуть более 6+ тысяч банкоматов (см. на карте ниже), разбросанных по городам России. По каждому банкомату имеется информацию: геоданные в виде широты, долготы и адреса, и принадлежность к группе.
    """)
    components.html(map_html, height=600, scrolling=True)
    st.markdown("""
    Основная работа для подготовки данных к обучению модели состояла в том, что парсить адреса на составляющие: населенный пункт, улица, номер дома, индекс.
    Сложность же задачи была в том, что адреса были представлены в муниципальном формате. В муниципальном делении указывают \
    структуры местного самоуправления, например, регион, районы, сельские и городские поселения, муниципальные округа. \
    """)
    st.markdown("""
    Ниже представлен график распределения топ-10 населенных пунктов по численности банкоматов.
    """)
    st.image("plot.jpg")
    st.markdown("""
    С промежуточным результатом, метриками качества, обученных моделей можно ознакомиться на графике (см. ниже).
    """)
    st.image("metrics.jpg")
    st.markdown("""
        Здесь Вам предлагается загрузить Ваши данные и получить прогноз.
        """)
    st.header("Загрузка данных")
    # Инициализация session_state
    if "dataframe" not in st.session_state:
        st.session_state["dataframe"] = None
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = None

    # Загрузка файла
    uploaded_file = st.file_uploader("Выберите CSV-файл", type=['csv'])
    if uploaded_file is not None:
        if st.button("Загрузить и обработать файл"):
            # Валидируем файл
            df, is_valid, message = validate_csv(uploaded_file)
            if is_valid:
                st.success(message)
                st.session_state["dataframe"] = df
                #display_data()  # Отображаем данные сразу после загрузки
            else:
                st.error(message)

    # Если данные уже загружены, отображаем их
    if st.session_state["dataframe"] is not None:
        display_data()

    # Показываем кнопку "Сделать прогноз" только если данные загружены
    if st.session_state["dataframe"] is not None:
        if st.button("Сделать прогноз"):
            df = st.session_state["dataframe"]
            # Отправляем файл на обработку в FastAPI
            csv_data = df.to_csv(index=False).encode('windows-1251')
            files = {"file": ("file.csv", csv_data)}
            response = requests.post(backend_url, files=files)

            if response.status_code == 200:
                st.success("Предсказания успешно выполнены!")
                # Загружаем результат
                result = pd.read_csv(io.StringIO(response.text))
                st.session_state["prediction_result"] = result
            else:
                st.error(f"Ошибка при отправке запроса: {response.status_code}")
                st.error(f"Ответ сервера: {response.text}")

    # Если есть предсказания, отображаем их
    if st.session_state["prediction_result"] is not None:
        st.write("Результаты предсказаний:")
        st.write(st.session_state["prediction_result"])
        st.download_button(
            label="Скачать результат",
            data=st.session_state["prediction_result"].to_csv(index=False).encode('utf-8'),
            file_name="predictions.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()