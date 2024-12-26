---
datasets:
- Paul/hatecheck-spanish
language:
- es
metrics:
- accuracy
- precision
- recall
- f1
base_model:
- dccuchile/bert-base-spanish-wwm-cased
pipeline_tag: text-classification
library_name: transformers
tags:
- bert
- transformer
- beto
- sequence-classification
- text-classification
- hate-speech-detection
- sentiment-analysis
- spanish
- nlp
- content-moderation
- social-media-analysis
- fine-tuned
---
# HateSpeech-BETO-cased-v1

<!-- Provide a quick summary of what the model is/does. -->

This model is fine-tuned version of [dccuchile/bert-base-spanish-wwm-cased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) for hate speech detection related to racism, homophobia, sexism, and other forms of discrimination in Spanish text.
It is trained on the dataset [Paul/hatecheck-spanish](https://huggingface.co/Paul/hatecheck-spanish).

## Metrics and results:

It achieves the following results on the *evaluation set* (last epoch):
- 'eval_loss': 0.03607647866010666
- 'eval_accuracy': 0.9933244325767691
- 'eval_precision_per_label': [1.0, 0.9905123339658444]
- 'eval_recall_per_label': [0.9779735682819384, 1.0]
- 'eval_f1_per_label': [0.9888641425389755, 0.9952335557673975]
- 'eval_precision_weighted': 0.9933877681310691
- 'eval_recall_weighted': 0.9933244325767691
- 'eval_f1_weighted': 0.9933031728530427
- 'eval_runtime': 1.7545
- 'eval_samples_per_second': 426.913
- 'eval_steps_per_second': 53.578
- 'epoch': 4.0

It achieves the following results on the *test set*:
- 'eval_loss': 0.052769944071769714
- 'eval_accuracy': 0.9933244325767691
- 'eval_precision_per_label': [0.9956140350877193, 0.9923224568138196]
- 'eval_recall_per_label': [0.9826839826839827, 0.9980694980694981]
- 'eval_f1_per_label': [0.9891067538126361, 0.9951876804619827]
- 'eval_precision_weighted': 0.9933376164683867
- 'eval_recall_weighted': 0.9933244325767691
- 'eval_f1_weighted': 0.993312254486016

### Training Details and Procedure

## Main Hyperparameters:

- evaluation_strategy: "epoch"
- learning_rate: 1e-5
- per_device_train_batch_size: 8
- per_device_eval_batch_size: 8
- num_train_epochs: 4
- weight_decay: 0.01
- save_strategy: "epoch"
- lr_scheduler_type: "linear"
- warmup_steps: 449
- logging_steps: 10


#### Preprocessing and Postprocessing:

- Needed to manually map dataset creating the different sets: train 60%, validation 20%, and test 20%.
- Needed to manually map dataset's labels, from str ("hateful", "non-hateful") to int (1,0), in order to properly create tensors.
- Dynamic Padding through DataCollator was used.


## More Information [optional]

- Fine-tuned by Javier de la Rosa Sánchez.
- javier.delarosa95@gmail.com
- https://www.linkedin.com/in/delarosajav95/

### Framework versions

- Transformers 4.47.0
- Pytorch 2.5.1+cu121
- Datasets 3.2.0
- Tokenizers 0.21.0