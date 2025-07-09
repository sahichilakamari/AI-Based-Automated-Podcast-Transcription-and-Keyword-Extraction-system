FROM python:3.9
RUN apt-get update && apt-get install -y ffmpeg
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD streamlit run app.py --server.port=$PORT
