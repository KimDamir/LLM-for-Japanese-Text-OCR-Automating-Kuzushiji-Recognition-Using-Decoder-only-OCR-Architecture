from inspect import signature
import torch
from model import DTrOCRLMHeadModel
from config import DTrOCRConfig
from dtrocr.data import DTrOCRLMHeadModelOutput, DTrOCRModelOutput, DTrOCRProcessorOutput
import torch_tensorrt

print(DTrOCRLMHeadModelOutput)
print(DTrOCRModelOutput)
print(DTrOCRProcessorOutput)
torch.export.register_dataclass(DTrOCRLMHeadModelOutput, serialized_type_name='DTrOCRLMHeadModelOutput')
torch.export.register_dataclass(DTrOCRModelOutput, serialized_type_name='DTrOCRModelOutput')
torch.export.register_dataclass(DTrOCRProcessorOutput, serialized_type_name='DTrOCRProcessorOutput')
ocrModel = DTrOCRLMHeadModel(DTrOCRConfig())
ocrModel.cuda()                               
state_dict = torch.load(f'../src/api/model/Version2.pt')
print(list(signature(ocrModel.forward).parameters.keys()))
new_dict = {}
for key in state_dict.keys():
    new_key = key
    if '_orig_mod.' in key:
        new_key = new_key.replace('_orig_mod.', '')
        
    new_dict[new_key] = state_dict[key]

inputs = [
        torch_tensorrt.Input(shape=(1, 3, 32, 128)), 
        torch_tensorrt.Input(shape=(1, 1), dtype=torch.int64)
        ]
ocrModel.load_state_dict(new_dict)
ocrModel = torch_tensorrt.compile(ocrModel, inputs=inputs,
                                  ir='dynamo'
                                  )
print('Compilation finished!')
save_inputs = [
        torch.randn(1, 3, 32, 128).cuda(), 
        torch.randint(12000, (1, 1), dtype=torch.int64).cuda()
        ]
# ocrModel = torch.compile(ocrModel)
torch_tensorrt.save(ocrModel, f'../src/api/model/Version2.ep', inputs=save_inputs)