import argparse
from collections import Counter
import code
import os
import logging
import random
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import AdamW, WarmupLinearSchedule
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def acc_f1_pea_spea(preds, labels):
    acc_f1 = acc_and_f1(preds, labels)
    pea_spea = pearson_and_spearman(preds,labels)
    return {**acc_f1, **pea_spea}

class FAQProcessor(DataProcessor):
    
    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'train.csv'))
    
    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'dev.csv'))

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, path):
        df = pd.read_csv(path)
        examples = []
        titles = [str(t) for t in df['title'].tolist()]
        replies = [str(t) for t in df['reply'].tolist()]
        labels = df['is_best'].astype('int').tolist()
        for i in range(len(labels)):
            examples.append(
                InputExample(guid=i, text_a=titles[i], text_b=replies[i], label=labels[i]))
        return examples
    
    def prepare_replies(self, data_dir):
        train_file = os.path.join(data_dir, 'train.csv')
        dev_file = os.path.join(data_dir, 'dev.csv')

        train_df = pd.read_csv(train_file)
        dev_df = pd.read_csv(dev_file)
        replies = [str(t) for t in train_df['reply'].tolist()] + [str(t) for t in dev_df['reply'].tolist()]

        return replies

def train(args, train_dataset, model, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps 
        args.num_train_epochs = args.max_steps //(len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    no_decay = ['bias', 'LayerNorm.weight']
    ## any((True, False, False)) 只要一个为True, 结果就为True,
    ## if not any() ->没有一个为True.  nd in n for nd in ['bias','LayerNorm.weight']->表示 parameter 不是'bias','weight'
    ## 所以下面第一个是 非 bias,weight的parameters, 第二个只有bias, weight的parameters
    ## 对于非bias,LayerNorm.weight 的parameters, weight_decay 值为args.weight_decay
    ## 对于 bias, LayerNorm.weight 的parameters, weight_decay 值为0
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    logger.info('*****Running training*******')
    logger.info(' Num examples = %d', len(train_dataset))
    logger.info(' Num epochs = %d', args.num_train_epochs)
    logger.info(' Gradient Accumulation steps = %d', args.gradient_accumulation_steps)
    logger.info(' Total optimization steps = %d', t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch')
    set_seed(args)
    preds, logging_preds = None, None
    out_label_ids, logging_out_label_ids = None, None
    best_acc_f1 = 0.0
    train_loss_file = os.path.join(args.output_dir, args.log_path, 'train_loss_file.txt')
    eval_acc_file = os.path.join(args.output_dir, args.log_path, 'eval_acc_file.txt')

   
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'labels': batch[3]}
            outputs = model(**inputs)
            loss, logits = outputs[:2] ## crossEntropy loss. outputs: (loss), logits, (hidden_states), (attentions)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            logging_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step() 
                scheduler.step() # update learning rate schedule
                model.zero_grad()
                global_step += 1
            

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

            if out_label_ids is None:
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

            if logging_preds is None:
                logging_preds = logits.detach().cpu().numpy() 
            else:
                logging_preds = np.append(logging_preds, logits.detach().cpu().numpy(), axis=0)

            if logging_out_label_ids is None:
                logging_out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                logging_out_label_ids = np.append(logging_out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)                

            
            if global_step % args.logging_steps == 0:
                results = acc_f1_pea_spea(np.argmax(logging_preds, axis=1), logging_out_label_ids)
                
                with open(train_loss_file, 'a+') as writer:
                    writer.write("iteration: {}, lr: {}, loss: {}, results:{}\n".format(global_step, scheduler.get_lr()[0], 
                                logging_loss/(args.logging_steps * args.train_batch_size * args.gradient_accumulation_steps), results))
                
                logging_loss = 0.0
                logging_preds = None
                logging_out_label_ids = None       
                # code.interact(local=locals())
                if args.evaluate_during_training:
                    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
                    results = evaluate(args, eval_dataset, model, args.device, tokenizer)
                    with open(eval_acc_file, 'a+') as eval_writer:
                        eval_writer.write('iteration:{}, lr: {}, eval_loss:{}, result: {}\n'.format(global_step, scheduler.get_lr()[0],results[0], results[1]))
                    if results[1]['acc_and_f1'] > best_acc_f1:
                        best_acc_f1 = results[1]['acc_and_f1']
                        print('saving best model')
                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(os.path.join(args.output_dir, args.log_path))
                        tokenizer.save_pretrained(os.path.join(args.output_dir, args.log_path))
                        torch.save(args, os.path.join(args.output_dir, args.log_path, 'training_args_bert.bin'))
 

def evaluate(args, eval_dataset, model, device, tokenizer):
    model.eval()
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)

    tr_loss = 0.0
    global_step = 0
    preds = None
    out_label_ids = None
    epoch_iterator = tqdm(eval_dataloader, desc='Iteration')
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'labels': batch[3]}
            # if step == 0:
            #     print(inputs)
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            batch_preds = None
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()    
            global_step += 1

            batch_preds = logits.detach().cpu().numpy() 
            batch_out_label_ids = inputs['labels'].detach().cpu().numpy()
            if preds is None:
                preds = batch_preds
            else:
                preds = np.append(preds, batch_preds, axis=0)
            if out_label_ids is None:
                out_label_ids = batch_out_label_ids
            else:
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            
        total_loss = tr_loss / (global_step * args.train_batch_size)
        print("iteration: {}, loss: {}".format(global_step, total_loss))
        preds = np.argmax(preds, axis=1)
        results = acc_f1_pea_spea(preds, out_label_ids)
        print(total_loss, results)
    return (total_loss,results)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    processor = FAQProcessor()
    cached_features_file = "cached_{}_bert".format("dev" if evaluate else 'train')
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        # print(len(examples))
        features = convert_examples_to_features(
                                examples=examples,
                                tokenizer=tokenizer,
                                max_length=args.max_seq_length,
                                label_list=label_list,
                                output_mode='classification',
                                pad_on_left=False,
                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                pad_token_segment_id=0)
        logger.info('saving features into cached file %s', cached_features_file)
        torch.save(features, cached_features_file)
    
    '''
        InputExample:
            self.guid = guid
            self.text_a = text_a
            self.text_b = text_b
            self.label = label
        InputFeatures:
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.token_type_ids = token_type_ids
            self.label = label

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))
    '''
    ## convert tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features],dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)
    return dataset

def convert_single_example_to_features(example, tokenizer, max_length=512, 
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    
    feature = []

    inputs = tokenizer.encode_plus(
        example.text_a,
        example.text_b,
        add_special_tokens=True,
        max_length=max_length,
        truncate_first_sequence=True  # We're truncating the first sequence in priority
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)

    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

    # logger.info("*** Example ***")
    # logger.info("guid: %s" % (example.guid))
    # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    # logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
    # logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))

    feature=InputFeatures(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label=None)

    return feature

    

def predict(context, replies, tokenizer, model, label_list, args):
    model.eval()

    best_score = 0.0
    best_reply = None

    results = []
    for index, reply in enumerate(replies):
        example = InputExample(guid=0, text_a = context, text_b = [reply])
        feature = convert_single_example_to_features(example, tokenizer, max_length=512, 
                                      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True)
 

        
        # with torch.no_grad():
        all_input_ids = torch.tensor([feature.input_ids],dtype=torch.long).to(args.device)
        all_attention_mask = torch.tensor([feature.attention_mask ], dtype=torch.long).to(args.device)
        all_token_type_ids = torch.tensor([feature.token_type_ids ], dtype=torch.long).to(args.device)
        #dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
        
        inputs = {'input_ids': all_input_ids, 'attention_mask': all_attention_mask, 'token_type_ids': all_token_type_ids}

        outputs = model(**inputs)
        
        logits = outputs[0]
        ## label is None, so there  we got logits. 
        # logits.detach().cpu().numpy() 
        prob = np.argmax(logits.detach().cpu().numpy() )
        results.append(prob)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="directory containing the data")
    parser.add_argument("--output_dir", default="BERT_output", type=str, required=True,
                        help="The model output save dir")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run predict.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--max_seq_length", default=100, type=int, required=False, 
                        help="maximum sequence length for BERT sequence classificatio")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--train_batch_size", default=64, type=int, required=False,
                        help="batch size for train and eval")
    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--log_path', default=None, type=str, required=False)
    parser.add_argument('--model_dir', default=None, type=str, required=False)

    args = parser.parse_args()
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    set_seed(args)
    ## get train and dev data
    print('loading dataset...')
    processor = FAQProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    # config = BertConfig.from_pretrained('bert-base-chinese', cache_dir='./cache_down', num_labels=num_labels)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./cache_down',do_lower_case=True, tokenize_chinese_chars=True)

    if args.do_train:
        #'BERT2_output/lr1e6_epoch5_seq512_warm1/'
        config = BertConfig.from_pretrained(args.model_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.model_dir,do_lower_case=True, tokenize_chinese_chars=True)

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        # 

        ## 构建模型
        model =  BertForSequenceClassification.from_pretrained(os.path.join(args.model_dir, 'pytorch_model.bin'), config=config)
        # model =  BertForSequenceClassification.from_pretrained("./cache_down/pytorch_model.bin", config=config)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(args.device)
        
        # create folder saving log
        if not os.path.exists(os.path.join(args.output_dir, args.log_path)):
            os.makedirs(os.path.join(args.output_dir, args.log_path))
        else:
            for file in os.listdir(os.path.join(args.output_dir, args.log_path)):
                os.remove(os.path.join(args.output_dir, args.log_path, file))
        
        # training
        train(args, train_dataset, model, tokenizer)
    
    if args.do_eval:
        if  args.log_path == None:
            print('pls input pretrained model path: --log_path')
            return
        if os.path.exists(os.path.join(args.output_dir, args.log_path)):
            # evaluate on best model
            config = BertConfig.from_pretrained(os.path.join(args.output_dir, args.log_path), num_labels=num_labels)
            tokenizer = BertTokenizer.from_pretrained(os.path.join(args.output_dir, args.log_path),do_lower_case=True, tokenize_chinese_chars=True)
            model =  BertForSequenceClassification.from_pretrained(os.path.join(args.output_dir, args.log_path, 'pytorch_model.bin'), config=config)
            
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(args.device)
            
            eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

            results = evaluate(args, eval_dataset, model, args.device, tokenizer)
                            
            print('eval_loss:{}, result: {}\n'.format(results[0], results[1]))
    
    if args.do_predict:
        
        config = BertConfig.from_pretrained(os.path.join(args.output_dir, args.log_path), num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(os.path.join(args.output_dir, args.log_path),do_lower_case=True, tokenize_chinese_chars=True)
        model =  BertForSequenceClassification.from_pretrained(os.path.join(args.output_dir, args.log_path, 'pytorch_model.bin'), config=config)
        
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(args.device)



        # candidate_file = os.path.join(args.output_dir, 'reply_candidates.pickle')
        replies = processor.prepare_replies(args.data_dir)
        # if not os.path.isfile(candidate_file):
        #     replies, vecotrs = prepare_replies(train_df, model, device, tokenizer, args)
        #     pickle.dump([replies, vecotrs], open(candidate_file, 'wb'))
        # else:
        #     replies, vecotrs = pickle.load(open(candidate_file, 'rb'))
        
        while True:
            title = input('你的问题?\n')
            if len(title.strip()) == 0:
                continue
            title = [title]
            ret = predict(title, replies, tokenizer, model, label_list, args)
            print(ret)  
            # scores = cosine_similarity(x_rep, vecotrs)[0]
            # index = np.argmax(scores)
            # print('可能的答案时:', replies[index])




if __name__== "__main__":
    main()
