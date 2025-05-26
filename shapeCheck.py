import torch


past_key_values = tuple([None] * 12)
print(past_key_values.shape)
past_length = past_key_values[0][0].size(-2)
# position_ids = torch.arange(past_length, input_shape[1] + past_length, dtype=torch.long, device='cpu')
# position_ids = position_ids.unsqueeze(0)