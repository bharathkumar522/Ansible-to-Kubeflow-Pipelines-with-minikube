FROM omie12/image-class:5.0

RUN pip install -q pyyaml h5py && \
     rm -rf tensorflow     

COPY . .

ENV ip "X.X.X.X" # your ip Address here
ENV user "root"
ENV passwd "admin123"

CMD /usr/bin/git clone http://${user}:${passwd}@${ip}:30080/${user}/tensorflow.git && cd tensorflow  && ls  && /usr/bin/python3 ../tf/prediction.py	 
	 
