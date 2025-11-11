import json
from pathlib import Path
import tqdm
import multiprocessing as mp
from dataclasses import dataclass
from dtrocr.processor import DTrOCRProcessor
from dtrocr.config import DTrOCRConfig
from pathlib import Path
from torch.utils.data import Dataset
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


