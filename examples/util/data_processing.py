import json

def save_labels(dict, output_dir):
    with open(str(output_dir / 'labels.json'), 'w') as fp:
        json.dump(dict, fp)