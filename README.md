![](./assets/framework.png)

# SYNAPSE: SYmbolic Neural-Aided Preference Synthesis Engine
[![Static Badge](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2403.16689)
[![Static Badge](https://img.shields.io/badge/Paper-Proceedings_(soon)-blue)]()
[![Static Badge](https://img.shields.io/badge/Demo-Website-orange)](https://amrl.cs.utexas.edu/synapse/)

This is the official repository for AAAI 2025 paper **SYNAPSE: SYmbolic Neural-Aided Preference Synthesis Engine**. SYNAPSE addresses the problem of preference learning, which aims to align robot behaviors through learning user-specific preferences (e.g. “good pull-over location”) from visual demonstrations. Despite its similarity to learning factual concepts (e.g. “red door”), preference learning is a fundamentally harder problem due to its subjective nature and the paucity of person-specific training data. We address this problem using a novel framework which is a **neuro-symbolic** approach designed to efficiently learn preferential concepts from limited data. SYNAPSE represents preferences as neuro-symbolic programs – facilitating inspection of individual parts for alignment – in a domain-specific language (DSL) that operates over images and leverages a novel combination of **visual parsing, large language models, and program synthesis** to learn programs representing individual preferences. We perform extensive evaluations on various preferential concepts as well as user case studies demonstrating its ability to align well with dissimilar user preferences. Our method significantly outperforms baselines, especially when it comes to out-of-distribution generalization.

## Requirements
We've tested on following setup:
- ROS Noetic
- Ubuntu 20.04
- OpenAI API access

## Overview
SYNAPSE consists of three main components: (1) Concept Library Update, (2) Sketch Synthesis, (3) Parameter Synthesis. At a high-level, the code structure is as follows:
- `src/sketch_synth.py`: Processes the NL descriptions and executes steps 1 and 2 together (i.e., concept library update and sketch synthesis).
- `src/datagen.py`: Extracts required quantities from the physical demonstrations to be later used in parameter synthesis. It produces an "SSR" (Structured Symbolic Representation) json of the demonstrations.
- `src/param_synth.py`: Uses the SSR jsons and the program sketch, and synthesizes the parameters for the program using LDIPS (`third_party/pips`).
- `src/infer.py`: Dynamically (i.e., program is not hard coded, instead executes whatever gets produced from synthesis process in `seqn_filled_lfps_sketches.json`) runs the produced program on given RGB image and produces segmentation predictions.
- `src/llmgrop/`: Contains the codebase for generalization to the tabletop arrangement [LLM-GROP](https://arxiv.org/abs/2303.06247) task.

## Setup
1. Create a conda environment:
```
conda create -n synapse python=3.10
```
2. Activate the environment and install [torch](https://pytorch.org/get-started/locally/). Then install dependencies:
```
bash setup.sh
```
3. Download demonstration bags from [here](https://drive.google.com/drive/folders/1pjAuhh4DQQNz1nUR8Fij4cuOAOQ_GdBU?usp=sharing) and place it in `test/demonstrations`.
4. Set the `OPENAI_API_KEY` in environment variables, e.g., add to your `.bashrc`:
```
export OPENAI_API_KEY=<your_openai_api_key>
```

## Usage
1. Run the following command to update the concept library and synthesize sketches:
```
python3 src/sketch_synth.py
```
2. Run the following command to extract the required quantities from the demonstrations:
```
python3 src/datagen.py
```
3. Run the following command to synthesize the parameters for the program:
```
python3 src/param_synth.py
```
4. Run the following command to infer the program on the given RGB image (`test/000000.png`):
```
python3 src/infer.py
```

**Note:** In the above, we use actual lidar data instead of predicted depth. However, it can be easily replaced with predicted depth (see `src/viz/fast_infer.py`).

## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex
@article{modak2024synapse,
  title={SYNAPSE: SYmbolic Neural-Aided Preference Synthesis Engine},
  author={Modak, Sadanand and Patton, Noah and Dillig, Isil and Biswas, Joydeep},
  journal={arXiv preprint arXiv:2403.16689},
  year={2024}
}
```