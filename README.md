README

1. All three MCQA datasets are put in the folder "data" and to unzip the RACE data, run the following command: 

   `tar -xf RACE.tar.gz`

2. Download the srl model to data_preprocess folder. link: https://pan.baidu.com/s/1Y6iIbX6AvhUSU1Q4k01rjQ password: xasm 

3.  To get semantic role lables of the input sequences, use the following command in data_preprocess folder: 

   `python race_offline.py`

   `mctest_off.py`

   `dream_off.py`

3. Download the pre-trained language model and put it in the dir "bert_base". The link is  

   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin

   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json

   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

4. To train the BERT model, use the following command:

   `python main.py TASK_NAME bert_base BATCH_SIZE_PER_GPU GRADIENT_ACCUMULATION_STEPS`

   Here we explain each required argument in details:

   TASK_NAME: it can be a single task or multiple tasks. If a single task, the options are: dream, race,  mctest160, and mctest500. Multiple tasks can be any combinations of those above-mentioned single tasks. For example, if you want to train a multi-task model on the dream and race tasks together, then this variable should be set as "dream,race".

   bert_base: model would be initialized by the parameters stored in this directory.

   BATCH_SIZE_PER_GPU: Batch size of data in a single GPU.

   GRADIENT_ACCUMULATION_STEPS: How many steps to accumulate the gradients for one step of back-propagation.

   e.g. 

   `python main.py race bert_base 2 4`

   `python main.py race.dream bert_base 2,2 4`

5. To facilitate your use of this code, I provide the labeled data:

link: https://pan.baidu.com/s/1Y6iIbX6AvhUSU1Q4k01rjQ password: xasm 