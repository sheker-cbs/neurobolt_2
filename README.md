<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>


<h3 align="center">NeuroBOLT custom implemetation guide</h3>

  <p align="center">
    project_description
    <br />
    <a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## Project layout

```text
neuro_bolt-main/
├── main.py                         # Namespace config + subject select + run folders
├── run_neuro.sh                    # SLURM script (NEUROBOLT_SUBJECT / training mode)
├── submit_sub.sh                   # one sbatch per subject via NEUROBOLT_SUBJECT
├── engine.py
├── runtime.py                      # live (imported by main)
├── utils.py
├── optim_factory.py
├── requirements.txt
├── arch/                           # baseline MSS
│   ├── model.py
│   └── model_multiscale.py
├── models/                         # attention MSS
│   ├── model.py
│   ├── model_multiscale.py
│   └── new_layers.py
├── dataset_maker/
│   └── get_datasets.py
├── checkpoints/                    # labram-base.pth; log_baseline/ & log_attn/ run folders
├── logs_baseline/                  # SLURM .out/.err for the default baseline mode
├── logs_attn/                      # SLURM .out/.err when #SBATCH paths are switched
├── README.md                       # this file


```
---
## About this project

This project contains a custom implementation of the NeuroBOLT pipeline, which reconstructs fMRI BOLD signals from EEG data. This fork trains one model per Algermissen subject (intrascan). Attention mode adds pooling layers in `new_layers.py` that record which frequency band and channel contribute most during training. 
 
---

<p align="right">(<a href="#readme-top">back to top</a>)</p>







<!-- GETTING STARTED -->
## Installation
To be able to run this fork or the official NeuroBOLT, you need have a virtual environment and following packages installed, along with the packages in `requirements.txt`: 
```sh

conda create -n neurobolt python=3.9
conda activate neurobolt
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install tensorboardX
pip install -r requirements.txt

```

## Code setup

### 1. `main.py`

Purpose: `main.py` is the main entry point, responsible for 'orchestrating' the pipeline. When invoked, the script loads the data, builds the model,finetunes with LaBraM hyperparameters, trains epochs, evaluates and saves logs/checkpoints/plots. It doesn't contain any training logic by itself. Contains configuration object `args` with hardcoded parameters that can be customized. 

Training modes: `baseline` (imports `arch.model`, sum/mean multi-scale-spectral module) or `attention` (imports `models.model`, adds band/channel attention). Set via `NEUROBOLT_TRAINING_MODE` env var.

Output: artifacts (files the training run produces) saved under `checkpoints/log_baseline/` or `checkpoints/log_attn/` depending on mode.


| Mode | Env | Model import | Artifact root | SLURM logs |
|------|-----|--------------|---------------|------------|
| **baseline** (default) | `NEUROBOLT_TRAINING_MODE=baseline` or unset | `arch.model` | `checkpoints/log_baseline/` | `logs_baseline/` |
| **attention** | `NEUROBOLT_TRAINING_MODE=attention` | `models.model` (+ `new_layers`) | `checkpoints/log_attn/` | `logs_attn/` |

`main.py` chooses the import from `NEUROBOLT_TRAINING_MODE` (also accepts `attn` / `base`). Do **not** switch the mode by hand and leave mode/env mismatched: artifacts and MSS pooling must stay consistent.

---

## SLURM job submission

`main.py` selects subjects from `NEUROBOLT_SUBJECT` or `SLURM_ARRAY_TASK_ID`. If neither is set, the job trains every Algermissen subject. It does not read batch files, `NEUROBOLT_SUBJECTS`, `NEUROBOLT_CAMPAIGN_DIR`, or `NEUROBOLT_SKIP_COMPLETED`. Checkpoints are written under `checkpoints/log_baseline/runs/` or `checkpoints/log_attn/runs/`, depending on `NEUROBOLT_TRAINING_MODE`.

`#SBATCH --output/--error` in `run_neuro.sh` point at `logs_baseline/` to match the default mode. For attention, change those two lines to `logs_attn/` before `sbatch` (Slurm fixes them at submit time).

```sh
# One subject, baseline (default)
NEUROBOLT_SUBJECT=sub-001 sbatch --export=ALL run_neuro.sh

# One subject, attention
NEUROBOLT_TRAINING_MODE=attention NEUROBOLT_SUBJECT=sub-001 sbatch --export=ALL run_neuro.sh

# Same, via the helper (one job per subject id)
bash submit_sub.sh sub-001
NEUROBOLT_TRAINING_MODE=attention bash submit_sub.sh sub-001
```

### Training mechanics

1. `engine.py` turns an EEG batch into a loss. 

| Parameter | Role |
|-----------|------|
| `model`, `criterion` | Network + MSE loss |
| `data_loader` | Yields `(EEG, target)` batches |
| `optimizer`, `loss_scaler` | Step weights; AMP GradScaler wrapper from `runtime` |
| `ch_names` → `input_chans` | Via `utils.get_input_chans` before forward |
| `spec_chan` | Optional spectral channel subset (`None` on live Algermissen) |
| `update_freq` | Accumulate this many micro-batches before optimizer step |
| `lr_schedule_values`, `wd_schedule_values` | Per-iteration LR / weight-decay |
| `max_norm` / `args.clip_grad` | Optional grad clip inside scaler |
| `is_binary` | If true, unsqueeze target to `[B,1]` (single ROI) |
| `/100`, `T=200` | EEG scale and patch length (hard-coded in loop) |

---

2. `optim.py` constructs an optimizer and optional ViT layer-decay parameter groups for fine-tuning.

| Parameter | Role |
|-----------|------|
| `create_optimizer(args, model, …)` | Construct AdamW (default) / other opts from `args.opt` |
| `args.lr`, `args.weight_decay`, `args.opt_eps`, `args.opt_betas` | Optimizer hyperparameters |
| `args.opt` | Optimizer name (`'adamw'`) |
| `LayerDecayValueAssigner` | Maps layer id → LR scale from `args.layer_decay` |
| `get_parameter_groups` | Split decay / no_decay (+ per-layer LR scale) |
| `get_num_layer_for_vit` | Assign transformer block depth for decay |
| `skip_list` | Params without weight decay (e.g. bias, `pos_embed`) |

---

3. `utils.py` and `runtime.py` share the same code infrastructure. 


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>








<!-- LICENSE
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTACT
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



