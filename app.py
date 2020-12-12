from flask import Flask,render_template,url_for,abort, request
from flair.models import TextClassifier
from flair.data import Sentence


from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
import pandas as pd
import os
import numpy as np
import logging
import torch
# load the model from disk
#filename = 'nlp.model.pkl'
#clf = pickle.load(open(filename, 'rb'))
#cv=pickle.load(open('trans.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1)[:, None])
    return e_x / np.sum(e_x, axis=1)[:, None]


class InputFeatures(object):
    """
    A single set of features for the data.
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id, agree=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.agree = agree

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode='classification'):
    """
    Loads a data file into a list of InputBatch's. With this function, the InputExample's are converted to features
    that can be used for the model. Text is tokenized, converted to ids and zero-padded. Labels are mapped to integers.

    Parameters
    ----------
    examples: list
        A list of InputExample's.
    label_list: list
        The list of labels.
    max_seq_length: int
        The maximum sequence length.
    tokenizer: BertTokenizer
        The tokenizer to be used.
    mode: str, optional
        The task type: 'classification' or 'regression'. Default is 'classification'

    Returns
    -------
    features: list
        A list of InputFeature's, which is an InputBatch.
    """

    if mode == 'classification':
        label_map = {label: i for i, label in enumerate(label_list)}
        label_map[None] = 9090

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length // 4) - 1] + tokens[
                                                              len(tokens) - (3 * max_seq_length // 4) + 1:]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        segment_ids = [0] * len(tokens)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if mode == 'classification':
            label_id = label_map[example.label]
        elif mode == 'regression':
            label_id = float(example.label)
        else:
            raise ValueError("The mode should either be classification or regression. You entered: " + mode)

        agree = example.agree
        mapagree = {'0.5': 1, '0.66': 2, '0.75': 3, '1.0': 4}
        try:
            agree = mapagree[agree]
        except:
            agree = 0

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          agree=agree))
    return features


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, agree=None):
        """
        Constructs an InputExample
        Parameters
        ----------
        guid: str
            Unique id for the examples
        text: str
            Text for the first sequence.
        label: str, optional
            Label for the example.
        agree: str, optional
            For FinBERT , inter-annotator agreement level.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.agree = agree


def chunks(l, n):
    """
    Simple utility function to split a list into fixed-length chunks.
    Parameters
    ----------
    l: list
        given list
    n: int
        length of the sequence
    """
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def predict1(text, model, write_to_csv=False, path=None):
    """
    Predict sentiments of sentences in a given text. The function first tokenizes sentences, make predictions and write
    results.
    Parameters
    ----------
    text: string
        text to be analyzed
    model: BertForSequenceClassification
        path to the classifier model
    write_to_csv (optional): bool
    path (optional): string
        path to write the string
    """
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    sentences = sent_tokenize(text)

    label_list = ['positive', 'negative', 'neutral']
    label_dict = {0: 'Buy', 1: 'Sell', 2: 'Hold'}
    result = pd.DataFrame(columns=['sentence','prediction','sentiment_score'])
    for batch in chunks(sentences, 5):

        examples = [InputExample(str(i), sentence) for i, sentence in enumerate(batch)]

        features = convert_examples_to_features(examples, label_list, 64, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        with torch.no_grad():
            logits = model(all_input_ids, all_segment_ids, all_input_mask)
            logits = softmax(np.array(logits))
            sentiment_score = pd.Series(logits[:,0] - logits[:,1])
            predictions = np.squeeze(np.argmax(logits, axis=1))

            batch_result = {'sentence': batch,
                            #'logit': list(logits),
                            'prediction': predictions,
                            'sentiment_score':sentiment_score}
            
            batch_result = pd.DataFrame(batch_result)
            #print('batch_result',batch_result)
            #print('result',result)
            result = pd.concat([result,batch_result], ignore_index=True)

    result['prediction'] = result.prediction.apply(lambda x: label_dict[x])
    #if write_to_csv:
    #    result.to_csv(path,sep=',', index=False)

    return result

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		inputQuery = request.form['message']
		print('inputQuery',inputQuery)
		model = BertForSequenceClassification.from_pretrained('finbert-sentiment1', cache_dir=None, num_labels=3)
		result1 = predict1(str(inputQuery),model)
		sen=str(result1['sentence'][0])
		typ=str(result1['prediction'][0])
		scr=str(round(result1['sentiment_score'][0] * 100))
	return render_template('result.html',senti = sen,predic=typ,score=scr)



if __name__ == '__main__':
	app.run(debug=True)