import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
    generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    print(f'true:\t {generation_output["sequences"]}')

    inputs = torch.tensor([[15496,    11,   616,  3290,   318, 13779,   290,   220]], dtype=torch.int)
    masks = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int)
    o1 = model.generate(inputs, attention_mask=masks, return_dict_in_generate=True, output_scores=True)
    print(f'o1:\t {o1["sequences"]}')

    pad_token = 9999

    inputs = torch.tensor([[pad_token, pad_token, 15496,    11,   616,  3290,   318, 13779,   290,   220]], dtype=torch.int)
    masks = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int)
    o2 = model.generate(inputs, attention_mask=masks, return_dict_in_generate=True, output_scores=True, pad_token_id=pad_token, max_new_tokens=10)
    print(f'o2:\t {o2["sequences"]}')


    pad_token = 8888

    inputs = torch.tensor([[pad_token, pad_token, 15496,    11,   616,  3290,   318, 13779,   290,   220]], dtype=torch.int)
    masks = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int)
    o3 = model.generate(inputs, attention_mask=masks, return_dict_in_generate=True, output_scores=True, pad_token_id=pad_token, max_new_tokens=10)
    print(f'o3:\t {o3["sequences"]}')

    __import__('pdb').set_trace()
    print()

