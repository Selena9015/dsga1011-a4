import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

# Task prefix in the T5 text-to-text format (only used when preprocess=True)
TASK_PREFIX = 'translate English to SQL: '

class T5Dataset(Dataset):

    def __init__(self, data_folder, split, preprocess=False):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        
        # preprocess=False: baseline; preprocess=True: task prefix + "<" mapping (improved model)
        self.preprocess = preprocess
        
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        # T5's own decoder start token is the pad token (id 0)
        self.decoder_start_id = PAD_IDX
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_queries = load_lines(os.path.join(data_folder, f'{split}.nl'))
        # Encoder input: (optional task prefix +) question; tokenizer appends </s>
        prefix = TASK_PREFIX if self.preprocess else ''
        self.encoder_ids = [torch.tensor(tokenizer(prefix + q).input_ids) for q in nl_queries]

        self.decoder_inputs = []
        self.decoder_targets = []
        if split != 'test':
            sql_queries = load_lines(os.path.join(data_folder, f'{split}.sql'))
            for sql in sql_queries:
                if self.preprocess:
                    # "<" is not in the T5 vocabulary (it becomes <unk>), so map it to a plain word;
                    # it is mapped back to "<" after generation in train_t5.py
                    sql = sql.replace('<', 'lessthan')
                # Target: SQL tokens followed by </s>
                target = tokenizer(sql).input_ids
                # Teacher forcing: decoder input = start token + target shifted right by one
                decoder_input = [self.decoder_start_id] + target[:-1]
                self.decoder_inputs.append(torch.tensor(decoder_input))
                self.decoder_targets.append(torch.tensor(target))
    
    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
      # The test set has no SQL, so only the encoder input is returned
        if self.split == 'test':
            return (self.encoder_ids[idx],)
        return self.encoder_ids[idx], self.decoder_inputs[idx], self.decoder_targets[idx]


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Dynamic padding: pad each batch only to its own longest sequence
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, preprocess=False):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split, preprocess)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, preprocess=False):
    train_loader = get_dataloader(batch_size, "train", preprocess)
    dev_loader = get_dataloader(test_batch_size, "dev", preprocess)
    test_loader = get_dataloader(test_batch_size, "test", preprocess)
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x