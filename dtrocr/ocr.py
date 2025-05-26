import PIL.Image as Image
import io

def ocr(model, processor, byteArrayImage, num_beams):
    image = Image.open(io.BytesIO(byteArrayImage))
    
    inputs = processor(
        images=image, 
        texts='',
        return_tensors='pt'
    )
    
    model_output = model.generate(
        inputs, 
        processor,
        num_beams=num_beams
    )
    
    predicted_text = processor.tokeniser.decode(model_output[0], skip_special_tokens=True)
    
    return predicted_text