FROM ubuntu

WORKDIR /app
ADD . /app

RUN apt-get update 
RUN apt-get install -y python3 python3-pip

RUN pip3 install -r requirements.txt 

EXPOSE 2004

CMD ["python3", "editme.py", "serve"]






