from torch import nn
from transformers import Trainer, AutoTokenizer, AutoConfig, AutoModel, TrainingArguments
import wandb
from datasets import Dataset
import torch
import json

from modeling import ReCMLLM, ReCMLLMConfig

torch.autograd.set_detect_anomaly(True)

AutoConfig.register("ReCMLLM", ReCMLLMConfig)
AutoModel.register(ReCMLLMConfig, ReCMLLM)


def load_json(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj

class DataCollator:
    def __call__(self, examples): # List[examples] -> Dict[str:Any] (as inputs of compute_loss)
        batch = {} 
        print(f"data num:{len(examples)}")
        print(f"keys:{examples[0].keys()}")

        for k in examples[0].keys():
            # print(f"{k}:{examples[0][k]}")
            batch[k] = torch.stack([torch.tensor(example[k], dtype=torch.long) for example in examples]) 
        return batch


class ReCMLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        # print(f"compute_loss inputs:{inputs}")
        # print(f"input_ids.shape:{inputs["input_ids"].shape}; labels.shape:{inputs["labels"].shape}")
        bsz = inputs["input_ids"].size(0)
        output = model(inputs=inputs["input_ids"], labels=inputs["labels"], num_items_in_batch=torch.tensor([num_items_in_batch for _ in range(bsz)], dtype=torch.long))

        wandb.log({
                    "training_loss":output["loss"], 
                    "causal_lm_loss":output["causal_lm_loss"],
                    "cmp_loss":output["cmp_loss"]
                    })

        return (output["loss"], output) if return_outputs else output["loss"]

data = torch.load("mini_data.pt", weights_only=False)  # data: List[Dict[str,Tensor]]
dataset = Dataset.from_list(data[:128])
# 一个奇怪的现象:张量到这里会变成List
# print(data[0])
# print(dataset[0])

data_collator = DataCollator()




model_config = load_json("./ReCMLLM/config.json")
model_config = ReCMLLMConfig(**model_config)
print(model_config)
model = ReCMLLM(model_config)
print(model)


tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)

training_args = TrainingArguments(**model_config.training_args)

trainer = ReCMLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

trainer.train()