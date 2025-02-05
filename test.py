import torch
from dtrocr.config import DTrOCRConfig
from dtrocr.model import DTrOCRLMHeadModel
from dtrocr.processor import DTrOCRProcessor

from PIL import Image

path_to_image = "examples/preprocessed_data/text_renderer/images/000000000.jpg"  # path to image file
image = Image.open(path_to_image).convert('RGB')
config = DTrOCRConfig()
model = DTrOCRLMHeadModel(config)
processor = DTrOCRProcessor(DTrOCRConfig())
model = torch.compile(model)
model.load_state_dict(torch.load('models/Version1.pt'))
model.eval()        # set model to evaluation mode for deterministic behaviour


inputs = processor(
    images=image,
    texts='',
    return_tensors="pt"
)

model_output = model.generate(
    inputs=inputs, 
    processor=processor, 
    num_beams=1,    # defaults to 1 if not specified
    use_cache=True  # defaults to True if not specified
)

predicted_text = processor.tokeniser.decode(model_output[0], skip_special_tokens=False)
print('Predicted text: ', predicted_text)