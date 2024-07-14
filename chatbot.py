import torch
import torch.nn as nn
import torch.nn.functional  as F
import csv
import random
import re
import os
import unicodedata
import codecs
import itertools
import math
import json

from torch.jit import script, trace
from torch import optim
from io import open

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

corpus_name = 'movie-corpus'
corpus = os.path.join("data", corpus_name)

def printLines(file, n=10):
    with open(file, 'rb') as datafie:
        lines = datafie.readlines()
    for line in lines[:n]:
        print(line)

printLines(os.path.join(corpus, "utterances.jsonl"))