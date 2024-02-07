#!/bin/bash

HYDRA_FULL_ERROR=1 python src/train_dpo.py model=blank_model datasets=[imdb] loss=dpo loss.beta=0.1 exp_name=sentiment_controlled_env gradient_accumulation_steps=4 batch_size=32 eval_batch_size=32 sample_during_eval=false

