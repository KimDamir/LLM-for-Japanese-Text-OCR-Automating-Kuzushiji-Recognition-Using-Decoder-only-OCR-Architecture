import json
from pathlib import Path
import random
from random import shuffle
import numpy as np
import tqdm
import multiprocessing as mp
from dataclasses import dataclass
from dtrocr.processor import DTrOCRProcessor
from dtrocr.config import DTrOCRConfig
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data.dataloader import Sampler
from PIL import Image


def save_labels(dict, output_dir):
    with open(str(output_dir), 'w') as fp:
        json.dump(dict, fp)



dataset_path = Path('data')


@dataclass
class Word:
    file_path: Path
    transcription: str
    

def get_word(word_file):
    words = []
    words.append(
        Word(
            file_path=word_file[0],
            transcription=word_file[1]
        )
    )
    return words


def get_words_list(words_files):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        words_list = list(
            tqdm.tqdm(
                pool.imap(get_word, words_files), 
                total=len(words_files),
                desc='Building dataset'
            )
        )
        words = [word for words_array in words_list for word in words_array]
    random.seed(224)
    random.shuffle(words)
    print(words[0])
    return words
    
class WORDSDataset(Dataset):
    def __init__(self, words: list[Word], config: DTrOCRConfig):
        super(WORDSDataset, self).__init__()
        self.words = words
        self.processor = DTrOCRProcessor(config, add_eos_token=True, add_bos_token=True)
        
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, item):
        inputs = self.processor(
            images=Image.open(self.words[item].file_path).convert('RGB'),
            texts=self.words[item].transcription,
            padding='max_length',
            return_tensors="pt",
            return_labels=True,
        )
        return {
            'pixel_values': inputs.pixel_values[0],
            'input_ids': inputs.input_ids[0],
            'input_attention_mask': inputs.input_attention_mask[0],
            'label_attention_mask': inputs.label_attention_mask[0],
            'labels': inputs.labels[0]
        }

def split_dataset(DATASET_PATH):
    with open(str(DATASET_PATH / 'labels.json'), 'r') as fp:
        data_dict = json.load(fp)
    words_files = list(data_dict.items())
    shuffle(words_files)
    train_list, test_list = np.split(words_files, [int(len(words_files)*0.9)])
    train_dict, test_dict = {}, {}
    
    for items_list, res_dict in zip((train_list, test_list), (train_dict, test_dict)):
        for file, transcription in items_list:
            res_dict[file] = transcription
            
    save_labels(train_dict, str(DATASET_PATH / 'train_labels.json'))
    save_labels(test_dict, str(DATASET_PATH / 'test_labels.json'))
    print(f'Train size: {len(train_list)}; Test size: {len(test_list)}')
    

    
class WordsSampler(Sampler):
    def __init__(self, data, i=0, batch_size=32):
        self.seq = list(range(len(data)))[i * batch_size:]
    
    def __iter__(self):
        return iter(self.seq)
    
    def __len__(self):
        return len(self.seq)