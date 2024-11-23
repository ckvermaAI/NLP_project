# NLP_project

## Setup for starters_code

- Clone this repo and switch the working directory
```bash
git clone https://github.com/ckvermaAI/NLP_project.git
cd ./starters_code
```

- Install the requirements
```bash
pip install -r requirements.txt
```

- Run the model
```bash
# Training (checkpoints will be saved under output_dir)
python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/

# Evaluation
python3 run.py --do_eval --task nli --dataset snli --model ./trained_model/ --output_dir ./eval_output/
```

## Setup for universal_triggers

- Clone this repo and install requirements for universal_triggers
```bash
git clone https://github.com/ckvermaAI/NLP_project.git
# Install the allennlp which is present in this repo
pip install allennlp/
```

- Run the script
```bash
python universal_triggers/triggers.py
```


## Generating the triggers
```bash
# Use the hotflip attack to generate universal triggers

# 1) Attack on entailment class, to flip the label to contradiction
python triggers.py --label_filter entailment --target_label 1 2>&1 | tee hotflip/entailment-contradiction.log
# 2) Attack on entailment class, to flip the label to neutral
python triggers.py --label_filter entailment --target_label 2 2>&1 | tee hotflip/entailment-neutral.log

# 3) Attack on contradiction class, to flip the label to entailment
python triggers.py --label_filter contradiction --target_label 0  2>&1 | tee hotflip/contradiction-entailment.log
# 4) Attack on contradiction class, to flip the label to neutral
python triggers.py --label_filter contradiction --target_label 2 2>&1 | tee hotflip/contradiction-neutral.log

# 5) Attack on neutral class, to flip the label to entailment
python triggers.py --label_filter neutral --target_label 0 2>&1 | tee hotflip/neutral-entailment.log
# 6) Attack on neutral class, to flip the label to contradiction
python triggers.py --label_filter neutral --target_label 1 2>&1 | tee hotflip/neutral-contradiction.log
```



## Source code

### Starters code
Pulled from https://github.com/gregdurrett/fp-dataset-artifacts

### Universal triggers
Pulled from https://github.com/Eric-Wallace/universal-triggers/
AllenNLP code: https://github.com/allenai/allennlp/tree/v0.8.5