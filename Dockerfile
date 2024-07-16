FROM python:3.10-slim-buster

WORKDIR /app

copy . /app/

RUN pip install -r requirements.txt

CMD [ "streamlit", "run","app.py" ]
