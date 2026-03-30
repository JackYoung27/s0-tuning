from s0_tuning import S0Config, S0Trainer

config = S0Config(n_steps=20, lr=1e-3)
trainer = S0Trainer.from_pretrained("Qwen/Qwen3.5-4B", config=config)

# (full_text, prompt_token_length) pairs; prompt_length tokens are masked
prompts = ["Q: What is 2+2?\nA:", "Q: What is 3+5?\nA:"]
answers = [" 4", " 8"]
data = []
for prompt, answer in zip(prompts, answers):
    tokens = trainer.tokenizer(prompt)
    prompt_length = len(tokens["input_ids"])
    data.append((prompt + answer, prompt_length))

metrics = trainer.train(data)
print(f"Training done: {metrics}")

trainer.activate()
output = trainer.generate("Q: What is 7+3?\nA:")
print(f"Output: {output}")

trainer.save("./my_s0_states")

trainer2 = S0Trainer.from_pretrained("Qwen/Qwen3.5-4B")
trainer2.load("./my_s0_states")
trainer2.activate()
