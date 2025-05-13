# When Machines Write: A Statistical Approach to AI Text Detection

This repository contains the code and resources for our NeurIPS submission paper: "When Machines Write: A Statistical Approach to AI Text Detection".

## Repository Structure

- **Classifiers/**: Contains all classifier implementations used in the paper, including:
  - MPU (Multiscale Positive-Unlabeled)
  - PHC (Paired-Human Classifier)
  - TopicScore
  - TF-IDF
  - DetectGPT
  - RoBERTa-Finetuned
  - RoBERTa-Stylo

- **src/**: Core implementation code and utilities
- **cache_files/**: Cached model outputs and intermediate results
- **results/**: Experiment results and performance metrics

## Running Experiments

To reproduce the main experiments from our paper, run:

```bash
python exp_A.py
