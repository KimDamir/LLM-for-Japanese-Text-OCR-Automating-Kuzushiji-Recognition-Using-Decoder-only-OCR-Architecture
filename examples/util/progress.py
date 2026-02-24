from pathlib import Path
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
    with open('reports/' + 'report '+MODEL+' on '+DATASET+'.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Train loss', 'Train accuracy', 'Test loss', 'Test accuracy'])
        rows = zip(train_losses, train_accuracies, test_losses, test_accuracies)
        for row in rows:
            writer.writerow(row)
            
def write_class_report(MODEL, weighted_acc, acc, classes_count, filtered_weighted_acc):
    path = Path('class_accuracy/' + MODEL)
    path.mkdir(parents=True, exist_ok=True)
    with open('class_accuracy/' +MODEL+'/report.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Weighted accuracy:', 'Accuracy', 'Total classes', 'Filtered weighted accuracy'])
        rows = zip([weighted_acc], [acc], [classes_count], [filtered_weighted_acc])
        for row in rows:
            writer.writerow(row)