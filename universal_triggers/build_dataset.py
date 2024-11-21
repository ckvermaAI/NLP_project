from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk
from huggingface_hub import login
import random

# Replace with your Hugging Face token
token = "<token>"

# Log in to Hugging Face
login(token=token)


def add_word_to_hypothesis(example, word):
    if example["hypothesis"][0].isupper():
        example["hypothesis"] = example['hypothesis'][0].lower() + example["hypothesis"][1:]
    example["hypothesis"] = word + " " + example["hypothesis"]
    return example


def pick_random_elements(input_list):
    if len(input_list) < 10:
        raise ValueError("The input list must have at least 10 elements.")

    # Generate 10 unique random indices
    random_indices = random.sample(range(len(input_list)), 10)

    # Create a new list using the selected indices
    random_elements = [input_list[i] for i in random_indices]

    return random_elements


def create_dataset():
    # Load the SNLI dataset
    dataset = load_dataset("snli")

    sample_size_train = 30000
    sample_size_test = 10000
    sample_size_validation = 10000
    snli_example_train = dataset['train'].shuffle(seed=28).select(range(sample_size_train))
    snli_example_test = dataset['test'].shuffle(seed=28).select(range(sample_size_test))
    snli_example_validation = dataset['validation'].shuffle(seed=28).select(range(sample_size_validation))

    entailment = ["baldness", "symphony", "beef", "rocket", "meats", "aircraft", "roast", "qualifying", "soup",
                   "jupiter", "tortoise", "tortilla", "rain", "whale", "swing", "naps", "assaults", "diving", "breads",
                   "steaks", "bathtub", "deserted", "swimming", "sauce", "convicted", "grounder", "deadly", "pasta",
                   "backstroke"]
    contradiction = ["sleeps", "alien", "zombie", "zombies", "mars", "aliens", "mutant", "spaceship", "crocodiles",
                     "elephants", "monkeys", "cat", "cats", "elephant", "gorilla", "zebras", "shark", "gorillas",
                     "snakes", "powerball", "pokemon", "tyrannosaurus", "centauri"]
    neutral = ["capsizes", "nervous", "fearful", "angrily", "deathly", "dancing", "swimmers", "bathing", "hockey",
               "fertilizing", "grandmother", "sisters", "clones", "asteroid", "football", "bowling", "silent", "sledge",
               "siblings", "quidditch", "asleep", "cloned", "joyously", "supervillain", "businessman", "soccer",
               "bicycle", "touchdown", "princess", "sled", "shyly", "extraterrestrial", "cooking", "aunts", "dogs",
               "snowboarding", "fried", "moon", "ski"]

    # random_selection_entailement = pick_random_elements(entailment)
    # random_selection_contradiction = pick_random_elements(contradiction)
    # random_selection_neutral = pick_random_elements(neutral)

    entailment_selection = entailment[:10]
    contradiction_selection = contradiction[:10]
    neutral_selection = neutral[:10]

    selection = entailment_selection + contradiction_selection + neutral_selection
    # random.shuffle(random_selection)

    modified_datasets_train = []

    for idx, trigger in enumerate(selection):
        if trigger[0].islower():
            trigger = trigger[0].upper() + trigger[1:]
        selection[idx] = trigger

    chunk_size = 1000
    start_idx = 0
    end_idx = 1000
    for trigger in selection:
        chunk = snli_example_train.select(range(start_idx, end_idx))
        modified_chunk = chunk.map(lambda x: add_word_to_hypothesis(x, word=trigger))
        modified_datasets_train.append(modified_chunk)
        start_idx = end_idx
        end_idx = end_idx + 1000

    snli_example_train_modified = concatenate_datasets(modified_datasets_train)

    modified_datasets_test = snli_example_test.map(lambda x, index: add_word_to_hypothesis(x, word=selection[index % 30]), with_indices=True)

    modified_datasets_validation = snli_example_validation.map(lambda x, index_v: add_word_to_hypothesis(x, word=selection[index_v % 30]), with_indices=True)


    modified_dataset = DatasetDict({'train': snli_example_train_modified, 'test': modified_datasets_test,
                                    'validation': modified_datasets_validation})

    # Push the modified dataset to your repository
    repo_name = "architpka/snli_modified_dataset"
    modified_dataset.push_to_hub(repo_name)
    print("Dataset pushed to hugging face")


def main():
    print("Enter Main")
    create_dataset()

main()