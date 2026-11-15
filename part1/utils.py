import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
from functools import lru_cache

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# Nearest keys on a QWERTY keyboard, used to simulate realistic typos
QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh",
    "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol", "a": "qwsz", "s": "awedxz",
    "d": "serfcx", "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "j": "huikmn",
    "k": "jiolm", "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
    "b": "vghn", "n": "bhjm", "m": "njk",
}
SYNONYM_PROB = 0.3   # probability of replacing a word with a WordNet synonym
TYPO_PROB = 0.25     # probability of introducing a keyboard typo into a word
MIN_WORD_LEN = 4     # skip short words (e.g. "not", "no", "bad") to preserve the label

@lru_cache(maxsize=None)
def get_synonyms(word):
    # Collect single-word WordNet synonyms that differ from the original word
    synonyms = set()
    for synset in wordnet.synsets(word)[:1]:
        for lemma in synset.lemmas():
            candidate = lemma.name()
            if ("_" not in candidate and candidate.isalpha()
                    and len(candidate) >= MIN_WORD_LEN and candidate.lower() != word):
                synonyms.add(candidate.lower())
    return sorted(synonyms)  # sorted so random.choice is reproducible across runs

def match_case(new_word, original):
    # Keep the capitalization pattern of the original word
    if original.isupper():
        return new_word.upper()
    if original[0].isupper():
        return new_word.capitalize()
    return new_word


def add_typo(word):
    # Replace one random letter with a neighboring key on the keyboard
    positions = [i for i, ch in enumerate(word) if ch.lower() in QWERTY_NEIGHBORS]
    if not positions:
        return word
    i = random.choice(positions)
    new_char = random.choice(QWERTY_NEIGHBORS[word[i].lower()])
    if word[i].isupper():
        new_char = new_char.upper()
    return word[:i] + new_char + word[i + 1:]

def transform_word(match):
    # Apply synonym replacement and/or a typo to a single word
    word = match.group(0)
    if len(word) < MIN_WORD_LEN:
        return word
    # Synonyms only for lowercase words: skips names, titles and sentence-initial words
    if word.islower() and random.random() < SYNONYM_PROB:
        synonyms = get_synonyms(word)
        if synonyms:
            word = random.choice(synonyms)
    if random.random() < TYPO_PROB:
        word = add_typo(word)
    return word


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    
    # Synonym replacement + keyboard typos, applied word by word.
    # Only alphabetic words are touched, so punctuation and "<br />" tags are preserved.
    example["text"] = re.sub(r"[A-Za-z]+", transform_word, example["text"])

    ##### YOUR CODE ENDS HERE ######

    return example
