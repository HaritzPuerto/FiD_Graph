# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm

from src.options import Options
import src.slurm
import src.util
import src.data
import src.FiD_GCN
import src.conditionalqa_eval
import os
import json

import logging

def process_prediction(pred_str):
    answers = []
    for raw_answer in pred_str.split("Answer: ")[1:]:
        ans_txt = raw_answer.split(". Conditions: ")[0]
        list_cond = raw_answer.split(". Conditions: ")[1:]
        list_cond = [c.strip() for c in list_cond]
        if len(list_cond) == 1 and list_cond[0] == "NA":
            list_cond = []
        answers.append([ans_txt, list_cond])
    return answers

def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_f1, checkpoint_path):

    torch.manual_seed(opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=1,
        drop_last=True,
        num_workers=1,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(tqdm(train_dataloader)):
            step += 1
            (idx, labels, _, context_ids, context_mask, graph) = batch
            try:
                train_loss = model(
                    input_ids=context_ids.cuda(),
                    attention_mask=context_mask.cuda(),
                    labels=labels.cuda(),
                    graph=graph.to('cuda')
                )[0]

                train_loss.backward()

                if step % opt.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                curr_loss += train_loss.item()

                if step % opt.eval_freq == 0 and step > 0:
                    results, list_str_preds = evaluate(model, eval_dataset, tokenizer, collator, opt)
                    dev_f1 = results['total']['F1_with_conditions']
                    model.train()
                    if opt.is_main:
                        if dev_f1 > best_dev_f1:
                            best_dev_f1 = dev_f1
                            src.util.save(model, optimizer, scheduler, step, best_dev_f1,
                                    opt, checkpoint_path, 'best_dev')
                        log = f"{step} / {opt.total_steps} |"
                        log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                        log += f"evaluation: {100*dev_f1:.2f} total F1 w/ cond. |"
                        log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                        print(log)
                        logging.info(log)    
                        
                        curr_loss = 0.
                        path = os.path.join(checkpoint_path, "checkpoint")
                        epoch_path = os.path.join(path, f"step-{step}") #"step-%s" % step)
                        os.makedirs(epoch_path, exist_ok=True)
                        with open(epoch_path + "/predictions.txt", "w") as f:
                            json.dump(list_str_preds, f)
                        with open(epoch_path + "/results.json", "w") as f:
                            json.dump(results, f)

                if  step % opt.save_freq == 0:
                    src.util.save(model, optimizer, scheduler, step, best_dev_f1,
                            opt, checkpoint_path, f"step-{step}")
                if step > opt.total_steps:
                    break
            
            except Exception as e:
                logging.error(f"ERROR IN TRAINING IDX {idx}")
                logging.error(f"Graph: {graph}")
                raise Exception(e)
                

def evaluate(model, dataset, tokenizer, collator, opt):
    list_str_preds = []
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    dict_id2pred = {}
    dict_id2label = {}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            (idx, _, _, context_ids, context_mask, graph) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                graph=graph.to('cuda'),
                max_length=50
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                target = dataset.get_example(idx[k])["target"]
                dict_id2pred[dataset.get_example(idx[k])['id']] = process_prediction(ans)
                dict_id2label[dataset.get_example(idx[k])['id']] = process_prediction(target)
                list_str_preds.append(ans)
    
    results = src.conditionalqa_eval.calculate_eval_metrics(dict_id2pred, dict_id2label)
    return results, list_str_preds

if __name__ == "__main__":
    print("Start main")
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)
    print("Options parsed")
    torch.manual_seed(opt.seed)
    opt.is_main = True
    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    # logging = src.util.init_logging(
    #     checkpoint_path / 'run.log'
    # )
    
    logging.basicConfig(filename=os.path.join(checkpoint_path, 'run.log'), encoding='utf-8', level=logging.DEBUG)
    logging.info(f"logging started")
    model_name = 't5-' + opt.model_size
    model_class = src.FiD_GCN.FiDT5

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    # collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)
    collator = src.data.GraphCollator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        # global_rank=opt.global_rank, 
        # world_size=opt.world_size,
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    logging.info(f"train examples loaded")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1, drop_last=True, num_workers=1, collate_fn=collator)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        # global_rank=opt.global_rank,
        # world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)
    logging.info(f"eval examples loaded")
    t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
    model = src.FiD_GCN.FiDT5(t5.config)
    model.load_t5(t5.state_dict())
    model = model.to('cuda')
    optimizer, scheduler = src.util.set_optim(opt, model)
    step, best_dev_em = 0, 0.0

    model.set_checkpoint(opt.use_checkpoint)

    print("Start training")
    logging.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
