# ReCMLLM


##### 准备环境
```
git clone https://github.com/1azybug/ReCMLLM.git
cd ReCMLLM
conda create -n ReCMLLM python=3.10 -y
conda activate ReCMLLM
# [cuda12.1 用于 xformers安装] https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers
conda install pytorch==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
# (Optional) Testing the installation
python -m xformers.info

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

##### 准备数据
```
bash setup.sh
```

##### 数据处理
```
bash train.sh
```

