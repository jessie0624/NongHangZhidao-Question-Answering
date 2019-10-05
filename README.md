# QABot

This is an QABot base on 'nonghangzhidao_filter.csv' from https://github.com/SophonPlus/ChineseNlpCorpus 

The model are base on hunggingface's transformers https://github.com/huggingface/transformers

In this repo, the bert_main2.py is the main script, train.csv and dev.csv are the dataset by cutting nonghangzhidao.csv. 


If you want to repeat the repo, pls following steps:

1. pls install transformers first refering https://huggingface.co/transformers/installation.html

2. To make it avaliable to run, I change 2 functions in transformers: 
   'tokenization_utils.py' -  adding a line in convert_toknes_to_ids  when len(ids)>max_len,so that it can auto cutdown the input sequence. 
   'glue.py' - changing 'truncate_first_sequence' to False.
```python
    def convert_tokens_to_ids(self, tokens):
        """ Converts a single token, or a sequence of tokens, (str/unicode) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str) or (six.PY2 and isinstance(tokens, unicode)):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        if len(ids) > self.max_len:
            ids = ids[:self.max_len + 1] ## adding for sequence length  cutdown(添加了这一行,并且注释掉了下面3行)
            # logger.warning("Token indices sequence length is longer than the specified maximum sequence length "
            #                "for this model ({} > {}). Running this sequence through the model will result in "
            #                "indexing errors".format(len(ids), self.max_len))
        return ids
```
```python
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            truncate_first_sequence=False ##True changing True to False # We're truncating the first sequence in priority
        )
```
3. cmd to start:
```python
python bert_main2.py \
		--data_dir . \ ## 数据集的路径.我的train.csv和dev.csv都放到当前目录下了
		--output_dir BERT2_output \ ## 输出结果的路径
		--train_batch_size 4 \  
		--num_train_epochs 10 \
		--max_seq_length 512 \  ##这个值可以调整
		--warmup_steps 1 \      ## 这个值模型调优可以调,这个是learning rate 变化
		--learning_rate 1e-5 \  ## 初始learning_rate,建议设置小一些,因为我用默认5e-5时,模型loss一路上升,acc 一路下降,估计模型已经飘了起来
		--log_path lr1e5_epoch10_seq512_warm1  ## 这个是每次调参后保存的模型结果和train_loss_file.txt, eval_acc_file.txt,train_acc_file.txt. 方便后面对比
```
4. results and analysis

