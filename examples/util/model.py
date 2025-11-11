import matplotlib
import torch
import os
from dtrocr.config import DTrOCRConfig
from dtrocr.model import DTrOCRLMHeadModel
from torch.utils.data import DataLoader
from util.progress import write_report
from typing import Tuple
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dtrocr.processor import DTrOCRProcessor

def get_model(config, MODEL):
    torch.set_float32_matmul_precision('high')
    model = DTrOCRLMHeadModel(config)
    model = torch.compile(model)
    if os.path.exists(f'../models/{MODEL}.pt'):
        model.load_state_dict(torch.load(f'../models/{MODEL}.pt'))
    model.to(device=0)
    return model


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
    # set model to evaluation mode
    model.eval()
    
    losses, accuracies = [], []
    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader, total=len(dataloader), desc=f'Evaluating test set'):
            inputs = send_inputs_to_device(inputs, device=0)
            outputs = model(**inputs)
            
            losses.append(outputs.loss.item())
            accuracies.append(outputs.accuracy.item())
    
    loss = sum(losses) / len(losses)
    accuracy = sum(accuracies) / len(accuracies)
    
    # set model back to training mode
    model.train()
    
    return loss, accuracy

def send_inputs_to_device(dictionary, device):
    return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in dictionary.items()}


def train(model, train_dataloader, test_dataloader, EPOCHS, MODEL, LR, DATASET):
    use_amp = True
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    optimiser = torch.optim.Adam(params=model.parameters(), lr=LR)

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
            if (len(epoch_losses)%500000 == 0): 
                torch.save(model.state_dict(), f'../models/{MODEL}.pt')
                print(f"Train loss: {sum(epoch_losses) / len(epoch_losses)}, Train accuracy: {sum(epoch_accuracies) / len(epoch_accuracies)}")
            
            
        # store loss and metrics
        train_losses.append(sum(epoch_losses) / len(epoch_losses))
        train_accuracies.append(sum(epoch_accuracies) / len(epoch_accuracies))
        
        # tests loss and accuracy
        test_loss, test_accuracy = evaluate_model(model, test_dataloader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
                        
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
        
        print(test_word_record.transcription, ' ', predicted_text)
        plt.figure(figsize=(10, 5))
        plt.title(predicted_text, fontsize=24)
        plt.imshow(np.array(image, dtype=np.uint8))
        plt.xticks([]), plt.yticks([])
        plt.show()