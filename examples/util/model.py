import matplotlib
import torch
import os
from dtrocr.config import DTrOCRConfig
from dtrocr.model import DTrOCRLMHeadModel
from torch.utils.data import DataLoader
from util.progress import write_class_report, write_report
from typing import Tuple
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dtrocr.processor import DTrOCRProcessor
from pathlib import Path           
from shutil import copy

def get_model(config, MODEL):
    torch.set_float32_matmul_precision('high')
    model = DTrOCRLMHeadModel(config)
    model = torch.compile(model)
    if os.path.exists(f'../models/{MODEL}.pt'):
        model.load_state_dict(torch.load(f'../models/{MODEL}.pt'))
    model.to(device=0)
    return model


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, byclass_acc: bool = False) -> Tuple[float, float]:
    # set model to evaluation mode
    model.eval()
    
    losses, accuracies = [], []
    acc_by_class = {}
    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader, total=len(dataloader), desc=f'Evaluating test set'):
            inputs = send_inputs_to_device(inputs, device=0)
            outputs = model(**inputs)
            
            losses.append(outputs.loss.item())
            accuracies.append(outputs.accuracy.item())
    
    classes = inputs['labels'][:][1]
    loss = sum(losses) / len(losses)
    accuracy = sum(accuracies) / len(accuracies)
    
    # set model back to training mode
    model.train()
    
    return loss, accuracy

def send_inputs_to_device(dictionary, device):
    return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in dictionary.items()}


def train(model, train_dataloader: DataLoader, test_dataloader, EPOCHS, MODEL, LR, DATASET):
    use_amp = True
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    optimiser = torch.optim.Adam(params=model.parameters(), lr=LR)
    i = 0
    path_to_ind = Path(f'checkpoints/{MODEL} train_samp_ind.txt')
    if path_to_ind.exists():
        with open(f'{MODEL} train_samp_ind.txt', 'r') as f:
            i = int(f.readline())

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    for epoch in range(EPOCHS):
        epoch_losses, epoch_accuracies = [], []
        for inputs in tqdm.tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch + 1}'):
            
            # set gradients to zero
            optimiser.zero_grad()
            
            # send inputs to same device as model
            inputs = send_inputs_to_device(inputs, device=0)
            
            # forward pass
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                outputs = model(**inputs)
            
            # calculate gradients
            scaler.scale(outputs.loss).backward()
            
            # update weights
            scaler.step(optimiser)
            scaler.update()
            
            epoch_losses.append(outputs.loss.item())
            epoch_accuracies.append(outputs.accuracy.item())
            if (len(epoch_losses)%10000 == 0):
                with open(path_to_ind, 'w') as f:
                    f.write(str(len(epoch_losses) + i))
                if Path(f'../models/{MODEL}.pt').exists():
                    copy(f'../models/{MODEL}.pt', f'../models/{MODEL}_reserve.pt')
                torch.save(model.state_dict(), f'../models/{MODEL}.pt')
                print(f"Train loss: {sum(epoch_losses) / len(epoch_losses)}, Train accuracy: {sum(epoch_accuracies) / len(epoch_accuracies)}")
        i = 0    
            
        # store loss and metrics
        train_losses.append(sum(epoch_losses) / len(epoch_losses))
        train_accuracies.append(sum(epoch_accuracies) / len(epoch_accuracies))
        
        # tests loss and accuracy
        if test_dataloader is not None:
            test_loss, test_accuracy = evaluate_model(model, test_dataloader)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
        else:
            test_losses, test_accuracies = [0], [0]
                            
        print(f"Epoch: {epoch + 1} - Train loss: {train_losses[-1]}, Train accuracy: {train_accuracies[-1]}, Test loss: {test_losses[-1]}, Test accuracy: {test_accuracies[-1]}")
    
    write_report(MODEL, DATASET, train_losses, train_accuracies, test_losses, test_accuracies)

def test(model, tested_data, n_samples):
    matplotlib.rc('font', family='TakaoPGothic')
    test_processor = DTrOCRProcessor(DTrOCRConfig())
    for test_word_record in tested_data[:n_samples]:
        image_file = test_word_record.file_path
        image = Image.open(image_file).convert('RGB')
        
        inputs = test_processor(
            images=image, 
            texts='',
            return_tensors='pt'
        )
        
        model_output = model.generate(
            inputs, 
            test_processor,
            num_beams=1
        )
        
        predicted_text = test_processor.tokeniser.decode(model_output[0], skip_special_tokens=True)
        
        print('Ответ: ', test_word_record.transcription, ' Вывод модели: ', predicted_text)
        plt.figure(figsize=(10, 5))
        plt.title(predicted_text, fontsize=24)
        plt.imshow(np.array(image, dtype=np.uint8))
        plt.xticks([]), plt.yticks([])
        plt.show()

from jiwer import cer
def check_cer(model, test_data):
    model.eval()
    model.to('cpu')
    total_cer = 0.0
    test_processor = DTrOCRProcessor(DTrOCRConfig(attn_implementation = 'flash_attention_2'))
    for test_word_record in tqdm.tqdm(test_data, total=len(test_data), desc=f'Evaluating cer'):
        image_file = test_word_record.file_path
        image = Image.open(image_file).convert('RGB')
        
        inputs = test_processor(
            images=image, 
            texts='',
            return_tensors='pt'
        )
        
        model_output = model.generate(
            inputs, 
            test_processor,
            num_beams=1
        )
        
        predicted_text = test_processor.tokeniser.decode(model_output[0], skip_special_tokens=True)
        word_cer = cer(test_word_record.transcription.strip(), predicted_text.strip())
        total_cer += word_cer
    
    avg_cer = total_cer / len(test_data)
    return avg_cer

def check_class_accuracy(MODEL, model, test_data, to_exclude = []): #returns weighted_accuracy, class_map ([classes]=[entries_coount, accuracy])
    model.eval()
    model.to('cpu')
    weighted_acc = 0.0
    filtered_weighted_acc = 0.0
    filtered_acc = 0.0
    filtered_keys_count = 0
    filtered_samples_count = 0
    acc = 0.0
    class_map = {}  #[word] = [count, accuracy(correct guesses)]
    test_processor = DTrOCRProcessor(DTrOCRConfig()) #attn_implementation = 'flash_attention_2'
    for test_word_record in tqdm.tqdm(test_data, total=len(test_data), desc=f'Evaluating weighted and by-class accuracy'):
        word = test_word_record.transcription.strip()
        if (word not in class_map):
            class_map[word] = [0.0, 0.0]
        
        image_file = test_word_record.file_path
        image = Image.open(image_file).convert('RGB')
        
        inputs = test_processor(
            images=image, 
            texts=test_word_record.transcription,
            return_tensors='pt',
            padding='max_length',
            return_labels=True
        )
        
        model_output, accuracy = model.generate(
            inputs, 
            test_processor,
            num_beams=1
        )
        
        accuracy = accuracy.item()
        
        predicted_text = test_processor.tokeniser.decode(model_output[0], skip_special_tokens=True)
        class_map[word][0] += 1
        class_map[word][1] += accuracy #1 if word.strip() == predicted_text.strip() else 0
        
    classes_count = 0
    total_count = 0    
    for word in class_map.keys():
        acc += class_map[word][1]
        class_map[word][1] = class_map[word][1]/class_map[word][0]
        weighted_acc += class_map[word][1]
        if (word not in to_exclude):
            filtered_weighted_acc += class_map[word][1]
            filtered_acc += class_map[word][1] * class_map[word][0]
            filtered_keys_count += 1
            filtered_samples_count += class_map[word][0]
        classes_count += 1
        total_count += class_map[word][0]    
    weighted_acc = weighted_acc / len(class_map.keys())
    if len(to_exclude) > 0:
        filtered_weighted_acc = filtered_weighted_acc / filtered_keys_count
        filtered_acc = filtered_acc / filtered_samples_count
    acc = acc / total_count
    print(f"Balanced accuracy: {weighted_acc}, Accuracy: {acc}, Total classes: {classes_count}, Filtered balanced accuracy: {filtered_weighted_acc} , Filtered accuracy: {filtered_acc}")
    write_class_report(MODEL, weighted_acc, acc, classes_count, filtered_weighted_acc)
    return weighted_acc, class_map