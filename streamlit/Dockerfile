FROM python:3.10-slim


WORKDIR /app

COPY streamlit/requirements.txt .
RUN pip install -r requirements.txt

COPY streamlit/ .

ENTRYPOINT ["streamlit", "run", "streamlit/app_st.py", "--server.port=8501", "--server.address=0.0.0.0"]