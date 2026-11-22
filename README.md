# ROCOv2 text-to-image tests

Stable Diffusion 1.5 with LoRA, fine-tuned on ROCOv2 figure captions, and two image-only judges trained on separate
articles. The tests of MINIM's released code are in `minim-testcase/` (see its README).

## Folders

| Folder | Contents |
|---|---|
| `tools/` | shared code: data access, image features, splits, Stable Diffusion helpers |
| `prepare/` | joined table, ROCO-mini split, both judges, latents, and the model that test 1 redraws |
| `experiments/test1_setup/` | redraw check, judge check, split by article, training speed |
| `experiments/test2_concepts/` | which caption concepts an image checker can recognise |
| `experiments/test3_concepts/` | the same with larger checker sides and negation-aware labels |
| `experiments/test4_pilot/` | best of 4 picked by the reward-side checkers, scored by the judge side |

Shared files go to `data/` and `models/`. Each test writes to `runs/` in its folder; `results/` holds the outputs of
our run.

## Data

ROCOv2 v2.0.1: https://zenodo.org/records/10821435

Put the csv tables in `data/rocov2/tables/` and `train_images.zip`, `valid_images.zip` and `test_images.zip` in
`data/rocov2/zips/`. The images are read from the zips. The models (Stable Diffusion 1.5, sd-vae-ft-mse, DINOv2,
SwAV and the CLIP tokenizer) download on first use.

## Setup

```
pip install -r requirements.txt
```

## Run

From the repository root, in this order:

```
python prepare/build_master.py
python prepare/build_roco_mini.py
python prepare/build_judge1.py
python prepare/build_judge2.py
python prepare/prep_latents.py --res 256 --max-tier 1000
python prepare/train_lora.py --arm minim_paper --tier 1000 --steps 400
python prepare/generate.py --run minim_paper_t1000 --n-per-mod 100 --steps 15

python experiments/test1_setup/setup_check.py
python experiments/test1_setup/speed_test.py
python experiments/test2_concepts/concept_count.py
python experiments/test3_concepts/concept_fix.py
python experiments/test4_pilot/run_pilot.py
python experiments/test4_pilot/inspect_results.py
```
