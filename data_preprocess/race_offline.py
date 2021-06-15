from __future__ import absolute_import, division, print_function
import csv
import logging
import os
import sys
import string

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
    def __init__(self, input_ids, input_mask, segment_ids, len_seq_a, len_seq_b, input_tag_ids, orig_to_token_split_idx, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.len_seq_a = len_seq_a
        self.len_seq_b = len_seq_b
        self.input_tag_ids = input_tag_ids
        self.orig_to_token_split_idx = orig_to_token_split_idx
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
class RaceProcessor(DataProcessor):

    def get_train_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "train", level=level)

    def get_test_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "test", level=level)

    def get_dev_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "dev", level=level)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_dataset_name(self):
        return 'RACE'

    def _read_samples(self, data_dir, set_type, level=None):
        if level is None:
            data_dirs = ['{}/{}/{}'.format(data_dir, set_type, 'high'),
                         '{}/{}/{}'.format(data_dir, set_type, 'middle')]
        else:
            data_dirs = ['{}/{}/{}'.format(data_dir, set_type, level)]

        examples = []
        example_id = 0
        for data_dir in data_dirs:
            filenames = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
            for filename in filenames:
                with open(filename, 'r', encoding='utf-8') as fpr:
                    data_raw = json.load(fpr)
                    article = data_raw['article']
                    for i in range(len(data_raw['answers'])):
                        example_id += 1
                        truth = str(ord(data_raw['answers'][i]) - ord('A'))
                        question = data_raw['questions'][i]
                        options = data_raw['options'][i]
                        for k in range(len(options)):
                            guid = "%s-%s-%s" % (set_type, example_id, k)
                            option = options[k]
                            examples.append(
                                    InputExample(guid=guid, text_a=article, text_b=option, label=truth,
                                                 text_c=question))

        return examples
def post_allen(sentence):
    results = predictor.predict_batch_json(sentence)
    return results

def get_tags(tok_text):
    sent_tags = []
    sent_words = []
    for sent in tok_text:
        sen_verbs = sent['verbs']
        sen_words = sent['words']
        if len(sen_verbs) == 0:
            sen_tag = ["O"] * len(sen_words)
        else:
            sen_tag = sen_verbs[0]['tags']
        '''
        sent_words.append(sen_words)
        sent_tags.append(sen_tag)
        '''
        sent_words += sen_words
        sent_tags += sen_tag
    '''    
    print('***********sent_words**********:\n',len(sent_words),sent_words)
    print('***********sent_tags**********:\n',len(sent_tags),sent_tags) 
    '''
    return sent_words, sent_tags

def convert_examples_to_features(examples, label_list, max_seq_length,tokenizer,tag_tokenizer, is_multi_choice=True):

    label_map = {label: i for i, label in enumerate(label_list)}
    print('label map',label_map)
    n_class = len(label_list)

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
        tok_to_orig_index_a = []       
        tokens_a = []
        tokens_b = [] 
        tag_sequence_a = get_tags(tokens_a_tags)
        #token_tag_sequence_a = QueryTagSequence(tag_sequence[0], tag_sequence[1])
        tokens_a_org = tag_sequence_a[0] # words_tokenize
        tok_to_orig_index_a.append(0) # [cls]
        for (i, token) in enumerate(tokens_a_org):
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index_a.append(i + 1)
                tokens_a.append(sub_token)
        tok_to_orig_index_b = []
        token_tag_sequence_b = None
        if  tokens_b_tags: 
            tag_sequence_b = get_tags(tokens_b_tags)
            #token_tag_sequence_b = QueryTagSequence(tag_sequence[0], tag_sequence[1])
            tokens_b_org = tag_sequence_b[0]
            for (i, token) in enumerate(tokens_b_org):
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index_b.append(i)
                    tokens_b.append(sub_token)
            if len(tokens_a+tokens_b) > max_seq_length-3:
                print("too long!!!!",len(tokens_a+tokens_b), len(tokens_a),len(tokens_b))
                _truncate_seq_pair(tokens_a, tokens_b, tok_to_orig_index_a, tok_to_orig_index_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                print("too long!!!!", len(tokens_a))
                tokens_a = tokens_a[:(max_seq_length - 2)]
                tok_to_orig_index_a=tok_to_orig_index_a[:max_seq_length - 1] #already has the index for [CLS]
        tok_to_orig_index_a.append(tok_to_orig_index_a[-1] + 1)  # [SEP]
        over_tok_to_orig_index = tok_to_orig_index_a
        if  tokens_b_tags:
            tok_to_orig_index_b.append(tok_to_orig_index_b[-1] + 1)  # [SEP]
            offset = tok_to_orig_index_a[-1]
            for org_ix in tok_to_orig_index_b:
                over_tok_to_orig_index.append(offset + org_ix + 1)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        len_seq_a = tok_to_orig_index_a[len(tokens)-1] + 1  # word_length of article
        if  tokens_b_tags:
            tokens += tokens_b + ["[SEP]"]
            len_seq_b = tok_to_orig_index_b[len(tokens_b)] + 1  #+1 SEP -1 for index
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        pre_ix = -1
        start_split_ix = -1
        over_token_to_orig_map_org = []
        for value in over_tok_to_orig_index:
            over_token_to_orig_map_org.append(value)
        orig_to_token_split_idx = []
        for token_ix, org_ix in enumerate(over_token_to_orig_map_org):
            if org_ix != pre_ix:
                pre_ix = org_ix
                end_split_ix = token_ix - 1
                if start_split_ix != -1:
                    orig_to_token_split_idx.append((start_split_ix, end_split_ix))
                start_split_ix = token_ix
        if start_split_ix != -1:
            orig_to_token_split_idx.append((start_split_ix, token_ix))
        while len(orig_to_token_split_idx) < max_seq_length:
            orig_to_token_split_idx.append((-1,-1))
        '''
        #get the tag list        
        tag_ids_list_a = token_tag_sequence_a.convert_to_ids(tag_tokenizer)
        input_que_tag_ids = [1] + tag_ids_list_a[:len_seq_a - 2] + [2] #CLS and SEP
        if token_tag_sequence_b != None:
            tag_ids_list_b = token_tag_sequence_b.convert_to_ids(tag_tokenizer)
            doc_input_tag_ids = tag_ids_list_b[:len_seq_b - 1] + [2] #SEP
            input_tag_ids = input_que_tag_ids + doc_input_tag_ids
            while len(input_tag_ids) < max_seq_length:
                input_tag_ids.append(0)
            assert len(input_tag_ids) == max_seq_length
        else:
            input_tag_ids = [1] + tag_ids_list_a[:len_seq_a - 2] + [2] #CLS and SEP
            while len(input_tag_ids) < max_seq_length:
                input_tag_ids.append(0)
            assert len(input_tag_ids) == max_seq_length
        '''
        tag_ids_list_a = tag_tokenizer.convert_tags_to_ids(tag_sequence_a[1])
        input_que_tag_ids = [1] + tag_ids_list_a[:len_seq_a - 2] + [2] #CLS and SEP
        if token_tag_sequence_b != None:
            tag_ids_list_b = tag_tokenizer.convert_tags_to_ids(tag_sequence_b[1])
            doc_input_tag_ids = tag_ids_list_b[:len_seq_b - 1] + [2] #SEP
            input_tag_ids = input_que_tag_ids + doc_input_tag_ids
            while len(input_tag_ids) < max_seq_length:
                input_tag_ids.append(0)
            assert len(input_tag_ids) == max_seq_length
        else:
            input_tag_ids = [1] + tag_ids_list_a[:len_seq_a - 2] + [2] #CLS and SEP
            while len(input_tag_ids) < max_seq_length:
                input_tag_ids.append(0)
            assert len(input_tag_ids) == max_seq_length
 
        #print('#####input_tag_ids######',input_tag_ids)
        #print('******orig_to_token_split_idx*****:\n',orig_to_token_split_idx)
        #print('tok_to_orig_index_a',tok_to_orig_index_a,len(tok_to_orig_index_a),len(tokens_a_org))
        #print('*****tokens_a_org****:\n',token_tag_sequence_a)
        #datas = [{"sentence": sent} for sent in nlp(tokens)]
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
            features = [[]]
        else:
            features = []
        if is_multi_choice:
            features[-1].append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    len_seq_a = len_seq_a,
                    len_seq_b = len_seq_b,
                    input_tag_ids = input_tag_ids,
                    orig_to_token_split_idx=orig_to_token_split_idx,
                    label_id=label_id))
            if len(features[-1]) == n_class:
                features.append([])
        else:
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    len_seq_a = len_seq_a,
                    len_seq_b = len_seq_b,
                    input_tag_ids = input_tag_ids,
                    orig_to_token_split_idx=orig_to_token_split_idx,
                    label_id=label_id)) 
    return features
def _truncate_seq_pair(tokens_a, tokens_b, tok_to_orig_index_a, tok_to_orig_index_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            tok_to_orig_index_a.pop()
        else:
            tokens_b.pop()
            tok_to_orig_index_b.pop()

race = RaceProcessor()

max_seq_length = 512
data_dir = 'data/RACE'
predictor = Predictor.from_path('srl')
predictor._model = predictor._model.cuda()
train_f = os.path.join(data_dir, 'cached_train')
test_f = os.path.join(data_dir, 'cached_test')
dev_f = os.path.join(data_dir, 'cached_dev')
tokenizer = tokenization.BertTokenizer.from_pretrained('bert_base', do_lower_case=True)
tag_tokenizer = TagTokenizer()
labels = race.get_labels()

train_examples = race.get_train_examples(data_dir,'high')
train_features = convert_examples_to_features(train_examples, labels, max_seq_length,tokenizer,tag_tokenizer, is_multi_choice=True)
print("Saving features into cached file ",train_f)
torch.save(train_features, train_f)

test_examples = race.get_test_examples(data_dir,'high')
test_features = convert_examples_to_features(test_examples, labels, max_seq_length,tokenizer,tag_tokenizer, is_multi_choice=True)
print("Saving features into cached file ", test_f)
torch.save(test_features, test_f)

dev_examples = race.get_dev_examples(data_dir,'high')
dev_features = convert_examples_to_features(dev_examples, labels, max_seq_length,tokenizer,tag_tokenizer, is_multi_choice=True)
print("Saving features into cached file ", dev_f)
torch.save(dev_features, dev_f)
labels = race.get_labels()


