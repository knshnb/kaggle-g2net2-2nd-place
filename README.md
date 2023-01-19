# 2nd Place Solution of Kaggle G2Net2 Competition
This is the Preferred Wave's solution for [G2Net Detecting Continuous Gravitational Waves](https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves).

## Dataset
We have a single file `input/test_real.csv` that lists the test data with real noise, which we detected in a similar way as a [public notebook](https://www.kaggle.com/code/vslaykovsky/g2net-winning-strategy-with-external-data).
Please add the competition dataset under `input/`.
```
$ ls -F input
sample_submission.csv  test/  test_real.csv  train/  train_labels.csv
```

## Usage
Our solution does not require any training. You can make predictions for the test data by simply running the following one command.
```
python predict.py --data_name test --config_path config/default.yaml --seed 0 --out_dir result/seed0
```
It saves the results under `result/seed0/`. You can use `pred.csv` as a prediction. You can see the parameters of `--topk` (100 by default) highest scores for each data in `score.csv`.

By specifying `--data_name train`, you can run validation on train data.
```
python predict.py --data_name test --config_path config/default.yaml --seed 0 --out_dir result/seed0
```

It took around 20 seconds to predict single data on NVIDIA V100 (=around 3 GPU hours and 2 GPU days for the execution of all train data and test data, respectively).

## For higher scores
The prediction by the above command scores around 0.825 in the private leaderboard. Averaging the results of 2 seeds raises the score to around 0.828, which is enough to win 2nd place. You can increase the score to 0.832 by averaging more seeds (~5) and even to 0.836 by ensembling different configurations (`config/freq4.yaml` and `config/freq6.yaml`).

## Links
- For an overview of our key ideas and detailed explanation, please also refer to [2nd Place Solution: GPU-Accelerated Random Search](https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/discussion/376504) in Kaggle discussion.
