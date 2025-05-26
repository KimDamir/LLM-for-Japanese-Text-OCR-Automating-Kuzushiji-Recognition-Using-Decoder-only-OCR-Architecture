import json

def save_labels(dict, output_dir):
    with open(str(output_dir), 'w') as fp:
        json.dump(dict, fp)