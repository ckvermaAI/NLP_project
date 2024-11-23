from sklearn.neighbors import KDTree
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import load_archive
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.iterators import BasicIterator
import argparse
import attacks
import sys
import utils
sys.path.append('..')


def _parse_args():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=128, help='Universal_perturb_batch_size')
    parser.add_argument('--attack_method', type=str, default='hotflip_attack', choices=['hotflip_attack', 'random_attack', 'nearest_neighbor_grad'], help="Type of attack")
    parser.add_argument('--label_filter', type=str, default='entailment', choices=['entailment', 'neutral', 'contradiction'], help="Subsample the SNLI dataset to one class to do a universal attack on that class")
    parser.add_argument('--target_label', type=str, default='1', choices=['0', '1', '2'], help="The attack is targeted towards a specific class. '0': flip to entailment, '1': flip to contradiction, '2': flip to neutral")
    # No need to change
    parser.add_argument('--num_trigger_tokens', type=int, default=1, help="Number of tokens to prepend, for SNLI it should be 1")
    parser.add_argument('--dev_path', type=str, default="https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl", help="path to the development dataset")
    parser.add_argument('--model_path', type=str, default="https://allennlp.s3-us-west-2.amazonaws.com/models/esim-glove-snli-2019.04.23.tar.gz", help="path to the trained model")
    args = parser.parse_args()
    return args


def main(args):

    # Load SNLI dataset
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    tokenizer = WordTokenizer(end_tokens=["@@NULL@@"]) # add @@NULL@@ to the end of sentences
    reader = SnliReader(token_indexers={'tokens': single_id_indexer}, tokenizer=tokenizer)
    dev_dataset = reader.read(args.dev_path)
    
    # Load model and vocab
    model = load_archive(args.model_path).model
    model.eval().cuda()
    vocab = model.vocab

    # Add hooks for embeddings so we can compute gradients w.r.t. to the input tokens
    utils.add_hooks(model)
    embedding_weight = utils.get_embedding_weight(model) # save the word embedding matrix

    # Batches of examples to construct triggers
    iterator = BasicIterator(batch_size=args.batch_size)
    iterator.index_with(vocab)

    # Subsample the dataset to one class to do a universal attack on that class
    subset_dev_dataset = []
    for instance in dev_dataset:
        if instance['label'].label == args.label_filter:
            subset_dev_dataset.append(instance)

    # A k-d tree if you want to do gradient + nearest neighbors
    if args.attack_method == 'nearest_neighbor_grad':
        tree = KDTree(embedding_weight.numpy())

    # Get original accuracy before adding universal triggers
    utils.get_accuracy(model, subset_dev_dataset, vocab, trigger_token_ids=None, snli=True)
    model.train() # rnn cannot do backwards in train mode

    # Initialize triggers
    trigger_token_ids = [vocab.get_token_index("the")] * args.num_trigger_tokens
    
    # sample batches, update the triggers, and repeat
    for index, batch in enumerate(lazy_groups_of(iterator(subset_dev_dataset, num_epochs=args.num_epochs, shuffle=True), group_size=1)):
        # get model accuracy with current triggers
        utils.get_accuracy(model, subset_dev_dataset, vocab, trigger_token_ids, snli=True)
        model.train() # rnn cannot do backwards in train mode

        # get grad of triggers
        averaged_grad = utils.get_average_grad(model, batch, trigger_token_ids, args.target_label, snli=True)

        # find attack candidates using an attack method
        if args.attack_method == 'hotflip_attack':
            cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
                                                            embedding_weight,
                                                            trigger_token_ids,
                                                            num_candidates=40)
        elif args.attack_method == 'random_attack':
            cand_trigger_token_ids = attacks.random_attack(embedding_weight,
                                                        trigger_token_ids,
                                                        num_candidates=40)
        elif args.attack_method == 'nearest_neighbor_grad':
            cand_trigger_token_ids = attacks.nearest_neighbor_grad(averaged_grad,
                                                                embedding_weight,
                                                                trigger_token_ids,
                                                                tree,
                                                                100,
                                                                increase_loss=True)

        # query the model to get the best candidates
        trigger_token_ids = utils.get_best_candidates(model,
                                                      batch,
                                                      trigger_token_ids,
                                                      cand_trigger_token_ids,
                                                      snli=True)

if __name__ == '__main__':
    args = _parse_args()
    print(args)
    main(args)