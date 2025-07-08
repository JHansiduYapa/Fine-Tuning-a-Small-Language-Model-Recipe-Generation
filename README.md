---

# Fine-Tuning a Small Language Model for Cypher Query Generation

This project fine-tunes [Unsloth's Gemma-3 4B IT 4-bit model](https://huggingface.co/unsloth/gemma-3-4b-it-unsloth-bnb-4bit) to generate Cypher queries from natural language inputs. The training process leverages 4-bit quantization and LoRA for efficient learning on limited hardware.

---

## Objective

Train a compact and efficient language model that translates natural language questions into valid Cypher queries for Neo4j-style graph databases.

---

## Project Contents

* `notebooks` – Data preparation and model fine-tuning notebooks
* `baseline_outputs.csv` – Output from the baseline Gemini model
* `inference_outputs.csv` – Output from the fine-tuned model
* `inference_results_ht.csv` – Outputs from hyperparameter-tuned model variants

---

## Training Workflow

### 1. Model Loading & LoRA Configuration

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    load_in_4bit = True,
    token = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
```

### 2. Training with `SFTTrainer` from `trl`

```python
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
```

---

## Evaluation Metrics

| Metric  | Description                                     |
| ------- | ----------------------------------------------- |
| BLEU    | N-gram overlap between prediction and reference |
| ROUGE-L | Longest common subsequence matching             |

---

## Example Inference

```python
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

inputs = tokenizer(dataset[13]['text'], return_tensors="pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)

_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=512)
```

Expected Output:

```cypher
MATCH (t:Topic)
WHERE NOT t.label STARTS WITH "P"
RETURN DISTINCT t.label, t.description
```

---

## Requirements

Install required packages:

```bash
pip install unsloth transformers datasets trl peft accelerate bitsandbytes
```

---

## Highlights

* 4-bit quantized model for efficient memory usage
* LoRA tuning for parameter-efficient fine-tuning
* Native support for gradient checkpointing (`unsloth`)
* Fast training with TRL’s `SFTTrainer`

---

## License

MIT License

---
