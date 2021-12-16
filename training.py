import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import librosa

timit = load_dataset("timit_asr")
timit = timit.remove_columns(["word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

# save vocab_dict to a json file
import json
with open('vocab.json', 'r') as f:
    vocab_dict = json.load(f)
# Closing file
f.close()

def extract_all_phones(data):
  all_phones = []
  for i in range(data.num_rows):
    all_phones.extend(data[i]["phonetic_detail"]["utterance"])
  vocab = list(set(all_phones))
  return vocab

vocab = extract_all_phones(timit['train'])
vocab_test = extract_all_phones(timit['test'])
bos_token = '<s>'
eos_token = '</s>'
pad_token = '<pad>'
unk_token = '<unk>'
vocab.extend(vocab_test)
vocab.extend([bos_token, eos_token, pad_token, unk_token])
vocab = list(set(vocab))
vocab_dict = {v: k for k, v in enumerate(vocab)}

# save vocab_dict to a json file
import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", padding=True, truncation=True)
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, 
                                             padding_value=0.0, do_normalize=True, return_attention_mask=False,
                                             padding=True, truncation=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        temp_labels = processor(batch["phonetic_detail"]["utterance"]).input_ids
        batch["labels"] = [label for sentence_labels in temp_labels for label in sentence_labels]
    return batch

timit = timit.map(prepare_dataset, remove_columns=timit.column_names["train"], num_proc=4)

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

    # def __call__(self, input_features, label_features) -> Dict[str, torch.Tensor]:
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

from datasets import load_metric

wer_metric = load_metric("wer")

import numpy as np

# compute the phoneme error rate of the predicted compared to the ground truth
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # TODO: Need to fix this to recognize calc_PER
    # per = calc_PER.compute(predictions=pred_str, references=label_str)
    # return {"per": per}

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

from transformers import Wav2Vec2Model, Wav2Vec2Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configuration = Wav2Vec2Config(vocab_size=len(vocab_dict))
model = Wav2Vec2ForCTC(configuration)
model = model.to(device)

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir='training_output',
  group_by_length=True,
  per_device_train_batch_size=8,
  evaluation_strategy="steps",
  num_train_epochs=2,
  gradient_checkpointing=True, 
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=timit['train'],
    eval_dataset=timit['test'],
    tokenizer=processor.feature_extractor,
)

trainer.train()

# def map_to_result(batch):
#   with torch.no_grad():
#     input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
#     logits = model(input_values).logits

#   pred_ids = torch.argmax(logits, dim=-1)
#   batch["pred_str"] = processor.batch_decode(pred_ids)[0]
#   batch["text"] = processor.decode(batch["labels"], group_tokens=False)
  
#   return batch

# results = timit["test"].map(map_to_result, remove_columns=timit["test"].column_names)
