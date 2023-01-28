# Contrastive Sentence Representation Learning with Adaptive False Negative Cancellation

This repository contains code base for the paper "Contrastive Sentence Representation Learning with Adaptive False Negative Cancellation".

## Train SimCSE-AFNC

In the followsing section, we describe how to train a SimCSE-AFNC model by employing our code.

### Requirements
To fairhfully reproduced our resutls, you can use Pytorch 1.11.0 with CUDA11. Then run the following script to install the remaining dependencies,

`pip install -r requirements.txt`

In addition, you can choose proper top-k and screening strategy to obtain the better results.

### Training

#### Data
You can run data/download_wiki.sh to download the 1 million sentences from English Wikipedia as dataset for training.

#### Training scripts
--topk  choose the number of candiate false negatives, which is range from 1 to 4;
--phi   The similarity / difference-in-similarity threshold for screening false negative samples.
--screen_strategy  The strategy used for screen false negative; Strategy 1 is similarity strategy, while strategy 2 is difference-in-similarity strategy.

### Evaluation
Before evaluation, please download the evaluation datasets by running

`cd SentEval/data/downstream/` \
`bash download_dataset.sh`

Then come back to the root directory, you can evaluate the model using the following code:

`python evaluation.py `\
    `--model_name_or_path result/afnc-base_phi07_top3_str1_adelim ` \
    `--pooler cls ` \
    `--task_set sts ` \
    `--mode test`







