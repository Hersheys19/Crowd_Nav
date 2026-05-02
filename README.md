# Limited-MA-CrowdNav

We will focus on exapanding the codebase for HeR-DRL [[Code]](https://github.com/Zhouxy-Debugging-Den/HeR-DRL)[[Paper]](https://arxiv.org/pdf/2403.10083) to multiagent case.

This codebase should probably be very similar to CrowdNav_HEIGHT [[Code]](https://github.com/Shuijing725/CrowdNav_HEIGHT)[[Paper]](https://arxiv.org/pdf/2411.12150) due to hetregenious relational graphs used in the algorithm. The main difference is hetrogeniousity comes from other types of robots in the former and static objects in the later.

The CrowdNav_HEIGHT is built on top of the work CrowdNav++ [[Code]](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)[[Paper]](https://arxiv.org/pdf/2203.01821) from the same authors.

And finally all these works are built on top of the basic simulator CrowdNav [[Code]](https://github.com/vita-epfl/CrowdNav)[[Paper]](https://arxiv.org/pdf/1809.08835) at EPFL.

Our goal is to make a multi-agent CrowdNav with only few learning agents like they did in SAMARL [[Website]](https://sites.google.com/view/samarl/home) which in their case we can abviously figure out there only 3 agents in the environment not n agents.

To setup the conda environment (also compatible with cuda 12.4) do the following:
```
conda create -n CrowdNav3.8 python=3.8.20
conda activate CrowdNav3.8
conda install pip
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install numpy ==1.20.3 pandas==1.5.2 matplotlib==3.6.2
pip install gym==0.15.7
pip install cython
pip install tensorflow-gpu==2.11.0
```
To install OpenAI baselines:
```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```
Update numpy,
```
pip install numpy==1.21.6
```
To install RVO2
```
conda install -c conda-forge cmake
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
python setup.py build
python setup.py install
```
To run the original CrowdNav, you need to also do the following:
```
pip install gitpython
```
Make sure there is an empty `__init__.py` in configs folder. If it isn't there, create one. Modify the setup.py file to:
```
from setuptools import setup


setup(
    name='crowdnav',
    version='0.0.1',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.policy',
        'crowd_sim.envs.utils',
    ],
)
```
Then run:
```
pip install -e .
```


