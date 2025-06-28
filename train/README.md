# Table of Contents

- [Installation](#installation)  
  - [Install veRL](#install-verl)  
  - [Install Supported Environments](#install-supported-environments)  
    - [1. ALFWorld](#1-alfworld)  
    - [2. WebShop](#2-webshop)  
    - [3. Sokoban](#3-sokoban)  
    - [4. Gym Cards](#4-gym-cards)  
    - [5. APPWorld (Experimental)](#5-appworld-experimental)  
- [Run Examples](#run-examples)  
  - [RL Training](#rl-training)  
    - [1. GiGPO](#1-gigpo)  
    - [2. GRPO](#2-grpo)  
    - [3. PPO](#3-ppo)  
    - [4. RLOO](#4-rloo)  
    - [5. DAPO](#5-dapo)  
    - [6. GiGPO (dynamic)](#6-gigpo-dynamic)
  - [Qwen3](#qwen3)
  - [LoRA](#lora)
  - [Prompt-based Agent with GPT-4o](#prompt-based-agent-with-gpt-4o)


# Installation
## Install veRL
```bash
conda activate lmgame

# assume you are in the this `train` folder

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation

pip install -e .

pip install vllm==0.8.5
```

## Install Supported Environments
<!-- 
Details for installing each environment are provided in the [Environment Setup Guide](agent_system/environments/README.md).

`verl-agent` supports the following environments: **ALFWorld**, **WebShop**, **Gym Cards**, **Sokoban**, and **APPWorld** (experimental). -->

> ⚠️ **Important:** 
To run an agent in any of these environments, you must first install and configure the corresponding environment. We strongly recommend installing ***each environment in its own dedicated conda environment*** to avoid potential package version conflicts.

### 1. ALFWorld
Install with pip:
```bash
pip install gymnasium==0.29.1
pip install stable-baselines3==2.6.0
pip install alfworld
pip install vllm==0.8.5
```

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in `~/.cache/alfworld/`):
```bash
alfworld-download -f
```

Use `--extra` to download pre-trained checkpoints and seq2seq data.

Play a Textworld game:
```bash
alfworld-play-tw
```
---

### 2. WebShop
WebShop requires Python <=3.10, so begin by creating a new `verl-agent-webshop` environment
```bash
conda create -n verl-agent-webshop python==3.10 -y
conda activate verl-agent-webshop
```

Install WebShop
```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

Note: If you encounter issues with gdown, you may need visit `https://drive.google.com/`, get your Google Drive cookie, and paste it into `.cache/gdown/cookies.txt`.
Or you may need to manually download the files.


Verify that WebShop was installed correctly by running:
```bash
python run_web_agent_text_env.py
```

After WebShop is installed, return to the root directory of the repository and install the verl package in `verl-agent`:
```bash
cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.5
# spacy 3.7.2 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# weasel 0.3.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
```
The warnings can be safely ignored.

---
### 3. Sokoban
```bash
pip install matplotlib
pip install gym==0.26.2
pip install gym_sokoban==0.0.6
```
---
### 4. Gym Cards

```bash
cd repo_root/
pip3 install -e ./agent_system/environments/env_package/gym_cards/gym-cards/
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
```
---
### 5. APPWorld (Experimental)
Install APPWorld package in `verl-agent` (some warnings may be raised, you can ignore them)
```bash
cd repo_root/
cd ./agent_system/environments/env_package/appworld/appworld
pip install -e .
python -m appworld.cli install
appworld download data

cd repo_root/
appworld download data
```

Refresh dependencies in the `verl-agent` environment:
```bash
cd repo_root/
pip install -e .
pip install vllm==0.8.5
```
You can ignore the warning of incompatiblity for appworld, because we don't run appworld in `verl-agent` environment.

Create a Dedicated Conda Environment `appworld` for the APPWorld Server:
```bash
conda create -n appworld python=3.12 -y
conda activate appworld

cd ./agent_system/environments/env_package/appworld/appworld
pip install -e .
python -m appworld.cli install
```


<!-- > ⚠️ **Important:**  
To run an agent in any of these environments, you must first install and configure the corresponding environment. Please refer to the [Environment Setup Guide](agent_system/environments/README.md) for step-by-step installation instructions. -->

# Run Examples
## RL Training
We provide out-of-the-box scripts in the ["examples/"](./examples/) directory for training agents in different environments.
Start with `bash examples/gigpo_trainer/run_sokoban.sh` for lmgame.

Here are more examples:
### 1. GiGPO
GiGPO is our novel algorithm designed to support fine-grained credit assignment in long-horizon LLM agent training. It introduces a two-level grouping mechanism:
- Episode-level groups capture overall task success via total returns (like GRPO).
- Step-level groups gather repeated states across trajectories to compute relative advantages for individual actions.

GiGPO is fully critic-free, maintains the same GPU memory footprint and LLM rollout cost as GRPO, yet achieves significantly better training efficiency and performance.

```bash
bash examples/gigpo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/gigpo_trainer/run_webshop.sh # WebShop
```
```bash
bash examples/gigpo_trainer/run_sokoban.sh # Sokoban
```
### 2. GRPO
GRPO is a critic-free algorithm that estimates relative advantages based on a group of full episode trajectories.
```bash
bash examples/grpo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/grpo_trainer/run_webshop.sh # WebShop
```
### 3. PPO
PPO is a classic actor-critic algorithm that updates the policy using a clipped objective to ensure stable learning. It requires a separate value network (critic) to estimate state values.
```bash
bash examples/ppo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/ppo_trainer/run_webshop.sh # WebShop
```
### 4. RLOO
For RLOO, we use a leave-one-out estimate and the PPO-clip update (instead of the REINFORCE update), making it closer to [LOOP](https://arxiv.org/abs/2502.01600).
```bash
bash examples/rloo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/rloo_trainer/run_webshop.sh # WebShop
```
### 5. DAPO
DAPO enhances GRPO with techniques like dynamic sampling and clip-higher.
```bash
bash examples/dapo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/dapo_trainer/run_webshop.sh # WebShop
```
### 6. GiGPO (dynamic)
GiGPO uses dynamic sampling and clip-higher from DAPO
```bash
bash examples/gigpo_dynamic_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/gigpo_dynamic_trainer/run_webshop.sh # WebShop
```
## Qwen3
```bash
bash examples/gigpo_trainer/run_webshop_qwen3.sh
```

## LoRA
```bash
bash examples/gigpo_trainer/run_alfworld_lora.sh
```

## Prompt-based Agent with GPT-4o
We also provide a prompt-based GPT-4o agent.
```bash
bash examples/prompt_agent/run_gpt4o_agent.sh
```

# Contributing

We welcome and appreciate all contributions! If you have ideas to improve `verl-agent`, please feel free to submit a pull request (PR).

## 1. Add New Environments
To add a new environment, 
1. Create your environment package (gym-style interface and multi-process execution) in [agent_system/environments/env_package/](./agent_system/environments/env_package/)
2. Define the corresponding prompt files in [agent_system/environments/prompts](./agent_system/environments/prompts/). 
3. Register your new environment in [env_manager.py](./agent_system/environments/env_manager.py), following the structure defined by [EnvironmentManagerBase](./agent_system/environments/base.py#L19). 

For a reference implementation, see the webshop environment:
1. Environment package: [webshop package](./agent_system/environments/env_package/webshop)
2. Prompts: [webshop prompts](./agent_system/environments/prompts/webshop.py)
3. Environment Manager: [webshop env manager](./agent_system/environments/env_manager.py#L304)


# Tips

## 1. Data Preparation
We only use data preparation to indicate the modality, either "text" or "visual". For example, if the task is purely text-based, the data will just be an empty string "". If it involves visual input, it will be "<image>". As for agent input (including task instruction, observation and prompt), we follow the classical RL pipeline. That means the input of LLM agent comes from the environment's feedback through `env.reset()` and `env.step()`.

## 2. Customize Your Own Prompts  
We adopt a simple and minimal prompt format in our implementation. For example, in the WebShop environment:
```
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}. Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}. You are now at step {current_step} and your current observation is: {current_observation}. Your admissible actions of the current situation are: [{available_actions}].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
```
If you wish to further enhance or customize them, you can find and edit them in: [agent_system/environments/prompts](./agent_system/environments/prompts/).

## Acknowledgement

* [veRL](https://github.com/volcengine/verl) 
* [verl-agent](https://github.com/langfengQ/verl-agent)
* [RAGEN](https://github.com/RAGEN-AI/RAGEN) 
