
import pandas as pd
import re
import numpy as np
import tweepy
import time
import tqdm.auto as tqdm


CONSUMER_KEY=""
CONSUMER_SECRET=""
ACCESS_TOKEN=""
ACCESS_TOKEN_SECRET=""

def rehydrate(file_name=None):
    ids =pd.read_csv(file_name)["id"].astype(str).values

    client = tweepy.Client(
        consumer_key=CONSUMER_KEY,
        consumer_secret=CONSUMER_SECRET,
        access_token=ACCESS_TOKEN,
        access_token_secret=ACCESS_TOKEN_SECRET
    )
    
    
    batch_size = 100
    max_per_interval = 890
    max_time_per_interval_seconds = 16*60
    data = []
    num_requests = 0
    start_time = time.time()
    
    for i in tqdm.trange(0,len(ids),batch_size):
        
        
        
        if num_requests == max_per_interval:
            elapsed_time = time.time() - start_time
            sleep_time = max_time_per_interval_seconds - elapsed_time
            if sleep_time>0:
                print(f"sleeping for {elapsed_time}")
                time.sleep(sleep_time)
            start_time = time.time()
            num_requests = 0
        
        
        batch_ids = ",".join(ids[i:i+batch_size])
        out = client.get_tweets(batch_ids, expansions="author_id", user_auth=True)
        data.extend([dict(o) for o in out.data])
        num_requests += 1
        
    pd.DataFrame(data).to_json(f"{file_name}.jsonl",orient="records",lines=True)


def clean_tweet(tweet, allow_new_lines = False):
    #courtesy of huggingtweet
    tweet = tweet.replace('&amp;', '&')
    tweet = tweet.replace('&lt;', '<')
    tweet = tweet.replace('&gt;', '>')
    bad_start = ['http:', 'https:']
    for w in bad_start:
        tweet = re.sub(f" {w}\\S+", "", tweet)      # removes white space before url
        tweet = re.sub(f"{w}\\S+ ", "", tweet)      # in case a tweet starts with a url
        tweet = re.sub(f"\n{w}\\S+ ", "", tweet)    # in case the url is on a new line
        tweet = re.sub(f"\n{w}\\S+", "", tweet)     # in case the url is alone on a new line
        tweet = re.sub(f"{w}\\S+", "", tweet)       # any other case?
    tweet = re.sub(' +', ' ', tweet)                # replace multiple spaces with one space
    if not allow_new_lines:                         # TODO: predictions seem better without new lines
        tweet = ' '.join(tweet.split("\n"))
    return tweet.strip()


def drop_mentions(tweet):
    words = tweet.split(" ")
    return " ".join([w for w in words if not w.startswith("@")]).strip()

def get_length(tweet):
    return len(tweet.split(" "))

def boring_tweet(tweet, min_non_boring_words = 5):
    boring_stuff = ['http', '@', '#']
    not_boring_words = len([None for w in tweet.split() if all(bs not in w.lower() for bs in boring_stuff)])
    return not_boring_words < min_non_boring_words


def create_dataset(file_name, to_drop_mentions = True):
    # assume ends iwth jsonl
    if file_name.endswith(".jsonl"):
        data = pd.read_json(file_name,lines = True).fillna("")
        file_name = file_name.replace(".jsonl","")
    elif file_name.endswith(".csv"):
        data = pd.read_csv(file_name).fillna("")
        file_name = file_name.replace(".csv","")

    text = data.text.str.replace("RT ","").drop_duplicates()
    text = text.apply(clean_tweet)

    if to_drop_mentions:
        text = text.apply(drop_mentions)

    boring = text.apply(boring_tweet)

    text = text.loc[~boring].fillna("")

    print("length 95 percentile", text.apply(get_length).describe([.95]))
    print("length", len(text))

    #split test train
    train_mask = np.random.rand(len(text))<.9
    test_mask = ~train_mask

    file_name = file_name.replace(".jsonl","")

    text.loc[train_mask].to_csv(f"{file_name}_train.csv", index = False)
    text.loc[test_mask].to_csv(f"{file_name}_test.csv", index = False)


 

if __name__ == "__main__":

    # usage examples
    rehydrate("datasetname_ids.csv")
    create_dataset("datasetname_ids.jsonl")