from __future__ import absolute_import, division, print_function
import csv
import logging
import os
import sys
import string
import torch
from io import open
import json
from os import listdir
from os.path import isfile, join
import pytorch_pretrained_bert.tokenization as tokenization
from nltk.tokenize import sent_tokenize, word_tokenize
from allennlp.predictors import Predictor
from nltk.tokenize import sent_tokenize, word_tokenize
from data_process.datasets import SenSequence, DocSequence, QuerySequence, QueryTagSequence
from tag_model.tag_tokenization import TagTokenizer

logger = logging.getLogger(__name__)
class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, input_tag_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_tag_ids = input_tag_ids
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, remove_header=False):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            if remove_header:
                next(reader)
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
class DreamProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            data_dir, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_dir + "/{}.json".format(set_type), 'r') as f:
            data = json.load(f)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    text_a = '\n'.join(data[i][0])
                    text_c = data[i][1][j]["question"]
                    options = []
                    for k in range(len(data[i][1][j]["choice"])):
                        options.append(data[i][1][j]["choice"][k])
                    answer = data[i][1][j]["answer"]
                    label = str(options.index(answer))
                    for k in range(len(options)):
                        guid = "%s-%s-%s" % (set_type, i, k)
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=options[k], label=label, text_c=text_c))

        return examples

def post_allen(sentence):
    results = predictor.predict_batch_json(sentence)
    return results
def choose_tag_set(tag_sets):
    cnt_tag = 0
    tag_ix = 0
    for ix, tag_set in enumerate(tag_sets):
        cnt_tmp_tag = 0
        for tag in tag_set:
            if tag != 'O':
                cnt_tmp_tag = cnt_tmp_tag + 1
        if cnt_tmp_tag > cnt_tag:
            cnt_tag = cnt_tmp_tag
            tag_ix = ix
    chosen_tag_set = tag_sets[tag_ix]
    return chosen_tag_set
def get_tags(tok_text):
    sent_tags = []
    sent_words = []
    for sent in tok_text:
        sen_verbs = sent['verbs']
        sen_words = sent['words']
        sen_tag = []
        if len(sen_verbs) == 0:
            sen_tag = ["O"] * len(sen_words)
        else:
            #sen_tag = sen_verbs[0]['tags']
            for ix,tag_verb in enumerate(sen_verbs):
                se_tag = sen_verbs[ix]['tags']
                sen_tag.append(se_tag)
            sen_tag = choose_tag_set(sen_tag)
        '''
        sent_words.append(sen_words)
        sent_tags.append(sen_tag)
        '''
        sent_words += sen_words
        sent_tags += sen_tag
        
    #print('***********sent_words**********:\n',len(sent_words),sent_words)
    #print('***********sent_tags**********:\n',len(sent_tags),sent_tags) 
    return sent_words, sent_tags

def convert_examples_to_features(examples, label_list, max_seq_length,tokenizer,tag_tokenizer, is_multi_choice=True):

    label_map = {label: i for i, label in enumerate(label_list)}
    print('label map',label_map)
    n_class = len(label_list)
    if is_multi_choice:
        features = [[]]
    else:
        features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))
        token_a = example.text_a  # article
        token_b = example.text_c + ' ' + example.text_b   # question + option
        # get the predictors: allennlp predict
        data_a = [{"sentence":sent} for sent in sent_tokenize(token_a)]
        data_b = [{"sentence":sent} for sent in sent_tokenize(token_b)]
        tokens_a_tags = post_allen(data_a)
        tokens_b_tags = post_allen(data_b)
        # 
        tag_sequence_a = get_tags(tokens_a_tags)
        tag_sequence_b = get_tags(tokens_b_tags)
        tag_ids_list_a = tag_tokenizer.convert_tags_to_ids(tag_sequence_a[1])
        tag_ids_list_b = tag_tokenizer.convert_tags_to_ids(tag_sequence_b[1])
        # bert ids sequence
        tokens_a = tokenizer.tokenize(example.text_a.lower())
        tokens_b = tokenizer.tokenize(example.text_b.lower())
        tokens_c = tokenizer.tokenize(example.text_c.lower())
        tokens_b = tokens_c + tokens_b
        _truncate_seq_pair(tokens_a, tokens_b,tag_ids_list_a,tag_ids_list_b, max_seq_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #tag-ids sequence
        input_que_tag_ids = [1] + tag_ids_list_a + [2] #CLS and SEP
        doc_input_tag_ids = tag_ids_list_b + [2] #SEP
        input_tag_ids = input_que_tag_ids + doc_input_tag_ids
        while len(input_tag_ids) < max_seq_length:
            input_tag_ids.append(0)
        assert len(input_tag_ids) == max_seq_length
 
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if is_multi_choice:
            features[-1].append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    input_tag_ids = input_tag_ids,
                    label_id=label_id))
            if len(features[-1]) == n_class:
                features.append([])
        else:
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    input_tag_ids = input_tag_ids,
                    label_id=label_id)) 
    if len(features[-1]) == 0:
        features = features[:-1]
    return features
def _truncate_seq_pair(tokens_a, tokens_b, tag_ids_list_a, tag_ids_list_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            tag_ids_list_a.pop()
        else:
            tokens_b.pop()
            tag_ids_list_b.pop()

dream = DreamProcessor()

max_seq_length = 512
data_dir = 'data/dream'
predictor = Predictor.from_path('srl')
predictor._model = predictor._model.cuda()
train_f = os.path.join(data_dir, 'cached_train')
test_f = os.path.join(data_dir, 'cached_test')
dev_f = os.path.join(data_dir, 'cached_dev')
tokenizer = tokenization.BertTokenizer.from_pretrained('bert', do_lower_case=True)
tag_tokenizer = TagTokenizer()
labels = dream.get_labels()

train_examples = dream.get_train_examples(data_dir)
train_features = convert_examples_to_features(train_examples, labels, max_seq_length,tokenizer,tag_tokenizer, is_multi_choice=True)
print("Saving features into cached file ",train_f)
torch.save(train_features, train_f)

test_examples = dream.get_test_examples(data_dir)
test_features = convert_examples_to_features(test_examples, labels, max_seq_length,tokenizer,tag_tokenizer, is_multi_choice=True)
print("Saving features into cached file ", test_f)
torch.save(test_features, test_f)

dev_examples = dream.get_dev_examples(data_dir)
dev_features = convert_examples_to_features(dev_examples, labels, max_seq_length,tokenizer,tag_tokenizer, is_multi_choice=True)
print("Saving features into cached file ", dev_f)
torch.save(dev_features, dev_f)
labels = race.get_labels()


