gcloud compute tpus tpu-vm ssh node-1  --zone europe-west4-a

gcloud auth application-default login 
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --upgrade clu

# Install screen and create new screen
pip install screen
screen -S train

# get Joy-Lunkad/deepmind-research.git
git clone https://github.com/Joy-Lunkad/deepmind-research.git
cd deepmind-research/nfnets
pip3 install -r requirements.txt

# reload deepmind-research
cd ..
cd ..
rm -rf deepmind-research
git clone https://github.com/Joy-Lunkad/deepmind-research.git
cd deepmind-research/nfnets
clear

python3 experiment.py --config=experiment.py --config.data_dir='gs://aimimagenet' --logtostderr


# for tensorboard 

gcloud compute tpus  tpu-vm ssh node-1 --zone=europe-west4-a  --ssh-flag="-4 -L 9001:localhost:9001"

# doesnt work inside screens
TPU_LOAD_LIBRARY=0 tensorboard --logdir 'gs://aim-resnet-exp4/train' --port 9001













