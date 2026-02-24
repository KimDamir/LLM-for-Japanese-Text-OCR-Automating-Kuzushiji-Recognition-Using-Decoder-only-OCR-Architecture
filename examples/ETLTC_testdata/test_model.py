import matplotlib
from dtrocr.config import DTrOCRConfig
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dtrocr.processor import DTrOCRProcessor
from pathlib import Path
from util.model import get_model

def test(MODEL, image_path):
    config = DTrOCRConfig()
    matplotlib.rc('font', family='TakaoPGothic')
    model = get_model(config, MODEL)
    test_processor = DTrOCRProcessor(config)
    
    image_file = Path(image_path)
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
    
    print('Вывод модели: ', predicted_text)
    plt.figure(figsize=(10, 5))
    plt.title(predicted_text, fontsize=24)
    plt.imshow(np.array(image, dtype=np.uint8))
    plt.xticks([]), plt.yticks([])
    plt.show()