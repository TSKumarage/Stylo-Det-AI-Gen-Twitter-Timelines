# Dataset
Machine Generated Text in Social Media

## Overview
The dataset includes three different types of timelines (fully authentic, fully machine generated, and mixed authentic-machine generated timelines) with varying timeline lengths (1, 5, 10, 20, 25). The data is divided into task1 and task2 and training and test examples with no overlap between tweets across tje the train and test.

Instructions and code for dataset contruction: https://gitlab.semaforprogram.com/semafor/teams/ta4/str/twitter-nlg

### Task1 dataset
The naming convention of each file is `{split}_{dataset}_{model}_{timeline_length}.json`, where the parameters are:

- `{split}`: {'train', 'test'}
- `{dataset}`: {'covid', 'vaccine', 'climate'}
- `{model}`: {'human','gpt2','gpt2-medium','gpt2-large','EleutherAI-gpt-neo-1.3B'}
- `{timeline_length}`: {1, 5, 10, 20, 25}

### Task2 dataset
For developing the 'hacked'/mixed timeline files, we append 'human' to the model name and only include a timeline length of 25 such that the file parameters follow:

- `{split}`: {'train', 'test'}
- `{dataset}`: {'covid', 'vaccine', 'climate'}
- `{model}`: {'human-gpt2','human-gpt2-medium','human-gpt2-large','human-EleutherAI-gpt-neo-1.3B'}
- `{timeline_length}`: {25}

Therefore, there should be 150 files in total. 75 training files and 75 test files.

*Note: for "human" files, only “covid” topic contains timelines where all tweets are from the same user. For other topics, human timelines could be from different users.*


## Train/Test Set
The train/test set consists of 7,200 timeline examples in total:

- 5000 single-tweet timelines per domain per model
- 1000 five-tweet timelines per domain per model
- 500 ten-tweet timelines per domain per model
- 250 twenty-tweet timelines per domain per model
- 200 twenty-five-tweet timelines per domain per model
- 250 twenty-five-tweet mixed timelines per domain per model

## Example File Format

Each file contains a JSON structure, essentially a list of dictionaries. The fully authentic and machine generated timeline dictionaries have the following structure:

```
[
    {
        'tweets': [tweet1, ..., tweetN]
    },
    ...,
    {
        'tweets': [tweet1, ..., tweetN]
    }
]
```

The mixed timeline dictionaries also contain the index where the machine generation begins:

```
[
    {
        'tweets': [tweet1, ..., tweetN],
        'start_of_mg': idx
    },
    ...,
    {
        'tweets': [tweet1, ..., tweetN],
        'start_of_mg': idx
    }
]
```

## Output File Format

### Task1 Output

In task 1, the goal is to detect if a user's timeline is machine generated. For each test file (denoted by `test_{topic}_{model}_{timeline_length}.json`), we expect a `.csv` file with original filename (for example, `test_{topic}_{model}_{timeline_length}.csv`).

The file will be in the following format containing two columns:

```
index, llr_score
0, .342
1, .502
2, -.202
...
M, .203
```

`index` corresponds to the index of the timeline examples in the original json file. `llr_score` is the log-likelihood ratio score of detection, where llr_score > 0 signals a fully authentic example and llr_score < 0 signals a machine generated timeline.

### Task2 Output

In task 2, the goal is to detect if a user's timeline contains machine generated tweets. 

For each test file (denoted by `test_{topic}_human-{model}_{timeline_length}.json`), we expect a `.csv` file with original filename (for example, `test_{topic}_human-{model}_{timeline_length}.csv`).


#### Main Task
The file will be in the following format containing two columns:

```
index, llr_score
0, .342
1, .502
2, -.202
...
M, .203
```

As in the previous task, `index` corresponds to the index of the timeline examples in the original json file. However, the `llr_score` is the log-likelihood ratio score of mixed timeline detection, where llr_score > 0 signals a fully authentic example and llr_score < 0 signals a mixed timeline.

#### Bonus Task

The bonus task focuses on localizing the machine generated text, therefore the goal is to predict the point where machine generation begins.

The file will be in the following format containing two columns:

```
index, start_of_mg
0, 5
1, 5
2, 0
...
M, 10
```

As in the previous tasks, `index` corresponds to the index of the timeline examples in the original json file. However, `start_of_mg` is the predicted index where machine generation begins.
