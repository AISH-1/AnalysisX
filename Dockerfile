FROM python:3.11
WORKDIR /analysisx
COPY . /analysisx
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 3000
CMD python ./app.py