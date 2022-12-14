#add apt packages
add-apt-repository -y ppa:jonathonf/ffmpeg-4
apt update
apt install -y ffmpeg git git-lfs

#create env
conda create -n whisper_finetuning python=3.9
conda activate whisper_finetuning

# install conda packages
conda install -y jupyter scikit-learn numpy pandas matplotlib
conda install -y -c pyviz holoviews bokeh
conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

#install pip packages
python -m pip install -U pip
python -m pip install tensorflow

python -m pip install librosa \
soundfile \
git+https://github.com/huggingface/datasets \
git+https://github.com/huggingface/transformers \
"evaluate>=0.3.0" \
jiwer \
more-itertools \
"wandb>=0.13.6" \
bitsandbytes \
audiomentations \
pydub