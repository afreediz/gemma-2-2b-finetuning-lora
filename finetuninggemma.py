!pip install -q torch bitsandbytes datasets accelerate loralib

import torch
import torch.nn as nn

print(torch.__version__)
print(torch.cuda.is_available())

from huggingface_hub import notebook_login

notebook_login()

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import os, bitsandbytes as bnb

model_name = "google/gemma-2-9b-it"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

i = 5
for p in model.named_parameters():
  print(p)
  i -= 1
  if i < 0:
    break

for param in model.parameters():
  param.requires_grad = False
  if param.ndim == 1:
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x):
    return super().forward(x).to(torch.float32)

model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    """
  printing the number of trainable paramters in the model
  """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

message = "Be yourself; everyone else is already taken."
text_ids = tokenizer(message, return_tensors="pt").to(model.device)

with torch.no_grad():
    res = model.generate(
        **text_ids,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
    )

print(tokenizer.decode(res[0], skip_special_tokens=True))

"""## Fine tuning"""

pip install peft

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print_trainable_parameters(model)

lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

import transformers
from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")

data

data['train'][0]

def merge_columns(example):
  example['prediction'] = example['quote'] + " ->: " + str(example['tags'])
  return example

data['train'] = data['train'].map(merge_columns)
data['train']['prediction'][0]

data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)

data

lora_model

from peft import PeftModel

# Check 1
print("Is PeftModel:", isinstance(lora_model, PeftModel))

# Check 2 - correct import path for newer transformers
from transformers.utils import is_peft_available
print("PEFT available:", is_peft_available())

# Check 3 - replicate Trainer's exact check
from peft import PeftMixedModel
def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

print("Trainer _is_peft_model:", _is_peft_model(lora_model))

# Check 4 - what quantization attributes exist
base = lora_model.base_model.model
print("base type:", type(base))
print("has hf_quantizer:", hasattr(base, 'hf_quantizer'))
print("quantization_config:", getattr(base.config, 'quantization_config', 'NOT SET'))

# Check 5 - type chain
print("type(lora_model):", type(lora_model))

import importlib
import transformers.utils
import transformers.utils.import_utils as iu

# Force re-check of peft availability
iu._peft_available = None  # clear the cached result

# Monkey-patch is_peft_available to always return True
# since we've confirmed peft IS installed
import peft  # confirm it imports fine
transformers.utils.is_peft_available = lambda: True

# Also patch it in trainer's local scope
import transformers.trainer as trainer_module
trainer_module.is_peft_available = lambda: True

# Verify
from transformers.utils import is_peft_available
print("PEFT available now:", is_peft_available())

import transformers.trainer as trainer_module
from peft import PeftModel, PeftMixedModel

# Inject the missing names into trainer's module scope
trainer_module.PeftModel = PeftModel
trainer_module.PeftMixedModel = PeftMixedModel

print("Patched successfully")

model.config.use_cache = False

trainer = transformers.Trainer(
    model=lora_model,
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,   # effective batch = 4
        warmup_steps=10,
        max_steps=50,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,                       # Gemma2 prefers bf16 over fp16
        logging_steps=1,
        output_dir='outputs',
        optim="paged_adamw_8bit",        # needed for 8-bit quantized base
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

lora_model.push_to_hub("afreediz/gemma-2-9b-it-tagger-test-adapter",
                      commit_message = "Testing Lora Training method",
                      private=False)

"""## Inference"""

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "afreediz/gemma-2-9b-it-tagger-test-adapter"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                            return_dict = True,
                                            load_in_8bit = True,
                                            device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model,peft_model_id)

batch = tokenizer("“Be yourself; everyone else is already taken.” ->: ", return_tensors='pt').to(model.device)

#with torch.cuda.amp.autocast():
#  output_tokens = model.generate(**batch, max_new_tokens=50)

# USE ABOVE COMMENTED CODE IF BELOW LINE OF CODE IS NOT WORKING AS EXPECTED
output_tokens = model.generate(**batch, max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

