# QABot

This is an QABot base on 'nonghangzhidao_filter.csv' from https://github.com/SophonPlus/ChineseNlpCorpus 

The model are base on hunggingface's transformers https://github.com/huggingface/transformers

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
