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

inputs = processor(
    images=image,
    texts='Hello',
    return_tensors="pt",
    return_labels=True
)
print(inputs.input_attention_mask.shape)
print(inputs.label_attention_mask.shape)
print(inputs.labels.shape)
print(inputs.pixel_values.shape)
print(inputs.input_ids)

model = torch.compile(model)
model.load_state_dict(torch.load('models/Version2.pt'))
model.eval()        # set model to evaluation mode for deterministic behaviour
model_output = model.generate(
    inputs=inputs, 
    processor=processor, 
    num_beams=1,    # defaults to 1 if not specified
    use_cache=True  # defaults to True if not specified
)

predicted_text = processor.tokeniser.decode(model_output[0], skip_special_tokens=False)
print('Predicted text: ', predicted_text)
print(model_output.past_key_values.shape)

position_ids = inputs.input_attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(inputs.input_attention_mask == 0, 1)
if model_output.past_key_values:
    position_ids = position_ids[:, -inputs.input_ids.shape[1]:]
    
print(position_ids.shape)