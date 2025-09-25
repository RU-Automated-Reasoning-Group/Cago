### Installation

Build the conda environments:

```bash
conda env create -f environment.yml
```

And then, run:

```bash
pip install -e .
```


### Run Cago

Training Scripts:

```bash
python dreamerv2_demo/gc_main.py --configs Meta_Disassemble(environment name in config file) --gpu_index 0 --demo_search_strategy 21 --expl_behavior 'Demo_BC_Explorer' --train_every 1 --steps 1000000 --logdir "your logdir path"
```

Use the tensorboard to check the results.

```bash
tensorboard --logdir ~/logdir/your_logdir_name
```
