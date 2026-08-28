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
├── run_neuro.sh                    # SLURM script (batch file as $1)
├── prepare_subject_batches.sh      # write subject_batches/batch_NNN.txt
├── submit_all_subjects.sh          # dated run + sbatch per batch;
├── engine.py
├── runtime.py                      # live (imported by main)
├── utils.py
├── optim_factory.py
├── requirements.txt
├── plot_attention_attribution.py   # Fig.5-style topo/freq (attention ckpts)
├── scan_split_example.xlsx         # VU split sheet from original NeuroBOLT (optional)
├── arch/                           # baseline MSS
│   ├── model.py
│   └── model_multiscale.py
├── models/                         # attention MSS
│   ├── model.py
│   ├── model_multiscale.py
│   ├── new_layers.py
│   └── check.py
├── dataset_maker/
│   ├── get_datasets.py
│   └── preproc.py
├── checkpoints/                    # labram-base.pth; log_baseline/ & log_attn/ dated runs
├── logs_baseline/                  # SLURM .out/.err (not old flat logs/)
├── logs_attn/                      # SLURM .out/.err (attention)
├── subject_batches/                # generated batch lists 
├── README.md                       # this file


```
---
## About this project

This project contains custom implementation of NeuroBOLT pipeline, which reconstructs fMRI BOLD signals from EEG data. This fork adapts the original pipeline to the `Algermissen` dataset, allowing for both per-subject and cross-subject training, adds attention pooling layers in `new_layers.py` to output which frequency band and channel contributes the most to the training, and outputs topographical and frequency plots. 
 
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

`prepare_subject_batches.sh`, `submit_sub.sh` and `run_neuro.sh` handle discovering subjects, splitting them into batches, and submitting cluster jobs. 

```sh
prepare_subject_batches.sh  →  writes subject_batches/batch_NNN.txt
            ↓
submit_all_subjects.sh      →  sbatch's one job per batch file
            ↓
run_neuro.sh (the SLURM job) →  runs `python main.py` for its assigned subjects

```

### 1. `prepare_subject_batches.sh`
Purpose: scans the Algermissen data directory for subject IDs (from `{subject}_block*.npz` filenames), excludes the same skip set as `main.py` (`sub-004, sub-015, sub-025`), and splits the remaining subjects into batch files of a fixed size under `subject_batches/batch_NNN.txt` (one subject ID per line). It also records the batch size used in `.batch_size so submit_all_subjects.sh` can detect if you've changed it.

Usage: 
```sh
bash prepare_subject_batches.sh                       # default: 5 subjects/batch
bash prepare_subject_batches.sh --batch-size 5
SUBJECTS_PER_JOB=5 bash prepare_subject_batches.sh     # same, via env var
bash prepare_subject_batches.sh --batch-size 1         # one subject per file
bash prepare_subject_batches.sh sub-001 sub-007 sub-012  # explicit subject list, skip scan


ALGERMISSEN_DATA_ROOT=/path/to/data bash prepare_subject_batches.sh #overrides the input directory 
BATCH_DIR=./my_batches bash prepare_subject_batches.sh # overrides the output directory of the batches
```


### 2.`submit_sub.sh`

Purpose: submits SLURM jobs in batches created by `prepare_subject_batches.sh` by invoking `run_neuro.sh`. Supports rerunning and resuming the interrupted batch jobs. Contains `--dry_run` command to preview without submitting jobs.

Output: creates a folder that contains `checkpoints/log_baseline/YYYYMMDD-HHMMSS/` or `log_attn/...` with a `campaign_meta`.json, batch size, and subject list.

Usage: 

```sh
# Standard: batch all subjects, 5 per job, submit to SLURM
bash submit_all_subjects.sh

# Custom batch size
bash submit_all_subjects.sh --batch-size 5
SUBJECTS_PER_JOB=5 bash submit_all_subjects.sh

# One subject per job (max parallelism)
bash submit_all_subjects.sh --one-per-job

# Preview without submitting
bash submit_all_subjects.sh --dry-run

# Reuse previously-generated batch files instead of rescanning data
bash submit_all_subjects.sh --reuse-batches

# Only specific subjects
bash submit_all_subjects.sh sub-001 sub-007 sub-012
bash submit_all_subjects.sh --batch-size 3 sub-001 sub-002 sub-003 sub-005

# Attention-mode training (instead of default baseline)
NEUROBOLT_TRAINING_MODE=attention bash submit_all_subjects.sh

# Resume an incomplete campaign — only resubmits subjects missing a checkpoint
bash submit_all_subjects.sh --resume-incomplete
bash submit_all_subjects.sh --resume-incomplete --campaign /path/to/checkpoints/log_baseline/20260820-113000
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



