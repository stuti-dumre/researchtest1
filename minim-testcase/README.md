# MINIM code tests

Scripts that test the released code of MINIM (https://github.com/WithStomach/MINIM) at commit 7a2b2f7.

## Setup

```
git clone https://github.com/WithStomach/MINIM.git MINIM
git -C MINIM checkout 7a2b2f7
pip install -r requirements.txt
```

The scripts look for MINIM in `./MINIM`. Set `MINIM_DIR` if it is somewhere else.

## Run

```
python t01_example_data.py
python t02_train_smoke.py
python t03_metrics.py
python t04_rlhf.py
python t05_classifier.py
python t06_laptop_speed.py
python make_figures.py
```

Each script saves its output in `results/`. `make_figures.py` uses the output of `t04_rlhf.py` and saves the plots
in `figures/`.
