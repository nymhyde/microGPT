import os
import numpy as np
import tiktoken
from datasets import load_dataset   # using huggingface datasets

from tqdm import tqdm    # loading bar


# num of workers for .map() call
nproc = 12

# loading the dataset
dataset = load_dataset("openwebtext")
print("Download Complete !")

'''
- the dataset occupies ~52 GB in huggingface
- about 8M documents
- by default : only contains the train split
-
'''

# creating train - val split
split_dataset = dataset['train'].train_test_split(test_size=0.005, seed=44, shuffle = True)
split_dataset['val'] = split_dataset.pop('test')


# defining the encoding function for tokenization
enc = tiktoken.get_encoding('gpt2')

def tokenizer(data):
    ids = enc.encode_ordinary(data['text'])
    # adding < end of text > token
    ids.append(enc.eot_token)
    out = {'ids' : ids, 'len': len(ids)}

    return out


# tokenizing the dataset
tokenized = split_dataset.map(tokenizer, remove_columns=['text'],
                              desc = "Tokenizing the splits",
                              num_proc = nproc)


# concatenating all the ids in 'train' and 'val' dataset into
# one larger file '.bin' to be used later

for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    print(f"Writing {filename} .....")

    idx = 0

    for data in tqdm(dset):
        arr[idx : idx + data['len']] = data['ids']
        idx += data['len']

    arr.flush()


print(f"Bin Files created .. Move on to train the model now ..")
