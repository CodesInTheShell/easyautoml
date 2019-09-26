FROM ubuntu

# System requirements
RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  python3-pip \
  swig \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip then install dependencies
RUN pip3 install --upgrade pip
RUN curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt \
  | xargs -n 1 -L 1 pip3 install

COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn"  , "-b", "0.0.0.0:5000", "app:app"]
