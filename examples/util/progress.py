import torch
import csv

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)
    
def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

def write_report(MODEL, DATASET, train_losses, train_accuracies, test_losses, test_accuracies):
    with open('report '+MODEL+'on '+DATASET+'.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow('Train loss', 'Train accurace', 'Test loss', 'Test accuracy')
        writer.writerows(zip(train_losses, train_accuracies, test_losses, test_accuracies))