from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained(
    'gpt2-large', pad_token_id=tokenizer.eos_token_id)

sentence = "How to be come a fullstack developer. Please advise me?"
input_ids = tokenizer.encode(sentence, return_tensors='pt')

output = model.generate(input_ids, max_length=1000,
                        num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
print(tokenizer.decode(output[0], skip_tokens=True))