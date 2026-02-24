from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
import multiprocessing as mp
from dtrocr.config import DTrOCRConfig
from util.data_processing import get_words_list, WORDSDataset
from util.model import get_model, train

MODEL='ETLTC'

EPOCHS=1

LR=1e-4

DATASET_PATH = Path('preprocessed_data/text_renderer')

VERTICAL_DATASET_PATH = Path('preprocessed_data/text_renderer_vertical')

DATASET_PATH.mkdir(parents=True, exist_ok=True)


with open(str(DATASET_PATH / 'augmented_labels.json'), 'r') as fp:
    data_dict = json.load(fp)

tr_dict = dict()

for items in data_dict.items():
    tr_dict[str(DATASET_PATH / 'images' / items[0]) + '.jpg'] = items[1]

with open(str(VERTICAL_DATASET_PATH / 'augmented_labels.json'), 'r') as fp:
    data_dict = json.load(fp)

vertical_dict = dict()

for items in data_dict.items():
    vertical_dict[str(VERTICAL_DATASET_PATH / 'images' / items[0]) + '.jpg'] = items[1]
    
print(f"{len(tr_dict.items())} dict elements")

words_files = list(tr_dict.items()) + list(vertical_dict.items())
print(f"{len(words_files)} dict elements")
print(words_files[0])

with open(str(DATASET_PATH / 'test_labels.json'), 'r') as fp:
    data_dict = json.load(fp)

tr_dict = dict()

for items in data_dict.items():
    tr_dict[str(DATASET_PATH / 'images' / items[0]) + '.jpg'] = items[1]


with open(str(VERTICAL_DATASET_PATH / 'test_labels.json'), 'r') as fp:
    data_dict = json.load(fp)

vertical_dict = dict()

for items in data_dict.items():
    vertical_dict[str(VERTICAL_DATASET_PATH / 'images' / items[0]) + '.jpg'] = items[1]

print(f"{len(tr_dict.items())} dict elements")

test_word_files = list(tr_dict.items()) + list(vertical_dict.items())
print(f"{len(words_files)} dict elements")
print(test_word_files[0])

test_words = get_words_list(test_word_files)
train_words = get_words_list(words_files)
print(f'Train size: {len(test_words)}; Test size: {len(train_words)}')
print(train_words[0], test_words[0])

config = DTrOCRConfig(
    # attn_implementation='flash_attention_2'
)
train_data = WORDSDataset(words=train_words, config=config)
test_data = WORDSDataset(words=test_words, config=config)


train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=mp.cpu_count())
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=mp.cpu_count())

model = get_model(config, MODEL)

train(model, train_dataloader, test_dataloader, EPOCHS, MODEL, LR, 'Text Renderer')
torch.save(model.state_dict(), f'../models/{MODEL}.pt')