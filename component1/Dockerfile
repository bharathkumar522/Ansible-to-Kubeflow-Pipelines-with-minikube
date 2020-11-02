FROM tensorflow/tensorflow:nightly-py3-jupyter
RUN apt-get update &&\
    apt-get upgrade -y &&\
	apt-get install -y git && \
	pip install matplotlib
	
COPY . .
ENV ip "X.X.X.X" #Keep your IP address here
ENV user "root"
ENV passwd "admin123"

CMD /usr/bin/git config --global user.name "root" && \
/usr/bin/git config --global user.email "admin@example.com" && \
/usr/bin/git clone http://${user}:${passwd}@${ip}:30080/${user}/tensorflow.git && \
echo "ipaddress ${ip}" && \
/usr/bin/python3 ./tf/classification.py && cd tensorflow/ && cp ../mymodel.h5 . && \
echo "pulling model" && \
/usr/bin/git pull --allow-unrelated-histories && \
/usr/bin/git add mymodel.h5 && \
/usr/bin/git commit "-m" "My first model" && \
/usr/bin/git push http://${user}:${passwd}@${ip}:30080/${user}/tensorflow.git


