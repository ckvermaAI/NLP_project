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
pip install allennlp
```

- Run the script
```bash
python universal_triggers/triggers.py
```

## Source code

### Starters code
Pulled from https://github.com/gregdurrett/fp-dataset-artifacts

### Universal triggers
Pulled from https://github.com/Eric-Wallace/universal-triggers/
AllenNLP code: https://github.com/allenai/allennlp/tree/v0.8.5