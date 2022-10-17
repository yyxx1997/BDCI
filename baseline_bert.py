import argparse
import os
import time
import datetime
import json
import json
from collections import defaultdict
from pathlib import Path
import torch
import torch.distributed as dist
from contextlib import nullcontext
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
...
from models.model_bert import BaselineBert
from models.tokenization_bert import BertTokenizer
from models.tricks import compute_kl_loss
import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

# https://zhuanlan.zhihu.com/p/363670628 
# https://huggingface.co/docs/transformers/main_classes/trainer
# TODO: function need to fix
# 1.early stop
# 2.warm up 
# 3.logging/saving strategy
# 4.max_grad_norm √
# TODO:Post processing after each step of training
# 1. Change the settings for the next training step
# 2. Record training status: log and intermediate process files
def training_loop(model, model_without_ddp, train_loader, val_loader, test_loader, optimizer, tokenizer, lr_scheduler):

    max_epoch = config.schedular['epochs']
    batch_size_train = config.batch_size_train
    K = config.gradient_accumulation_steps
    warmup_steps = config.schedular['warmup_epochs'] * K
    logging_step = config.logging_step * K
    logging_strategy = config.logging_strategy
    ckpt_output_path = config.ckpt_output_path
    r_drop_rate = config.r_drop
    max_grad_norm = config.max_grad_norm
    metrics = config.metrics
    
    best_scores = defaultdict(lambda:None)
    total_step = 0
    warmup_iterations = warmup_steps * 100
    total_train_batch_size = batch_size_train * K * config.world_size

    metric_logger = utils.MetricLogger(logging=logger.info, delimiter=" - ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=5, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=5, fmt='{value:.6f}'))

    logger.info("Start training")
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader.dataset)}")
    logger.info(f"  Num Epochs = {max_epoch}")
    logger.info(f"  Instantaneous batch size per device = {batch_size_train}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {K}")
    start_time = time.time()

    for epoch in range(1, max_epoch+1):
        logger.info(" -" * 20 + "Start of [{}/{}]".format(epoch, max_epoch) + " - " * 20)
        header = 'Train Epoch: [{}/{}]'.format(epoch, max_epoch)
        if config.distributed:
            train_loader.sampler.set_epoch(epoch)

        for i, batch in enumerate(metric_logger.log_every(train_loader, header=header)):
            
            model.train()
            batch = utils.prepare_input(batch, device)
            
            # Gradient Accumulation and Speed Up with No Sync: https://zhuanlan.zhihu.com/p/250471767
            my_context = model.no_sync if config.local_rank != -1 and i % K != 0 else nullcontext
            with my_context():
                output = model(batch)
                loss = output['loss']
                prediction = output['prediction']
                if r_drop_rate:
                    output_hat = model(batch)
                    loss_hat = output_hat['loss']
                    prediction_hat = output_hat['prediction']
                    ce_loss = 0.5 * (loss + loss_hat)
                    kl_loss = compute_kl_loss(prediction, prediction_hat)
                    loss = ce_loss + r_drop_rate * kl_loss
                loss = loss / K
                loss.backward()

            if i % K == 0:
                # https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
                # https://blog.csdn.net/zhaohongfei_358/article/details/122820992 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()

            total_step += 1
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(loss=loss.item())
            metric_logger.synchronize_between_processes()
            need_tb_logs = metric_logger.latest_meter(prefix='train/')

            if epoch == 1 and i % 100 == 0 and i <= warmup_iterations:
                lr_scheduler.step(i//100)

            if (logging_strategy == "epoch" and i == len(train_loader) - 1) or (logging_strategy == "steps" and total_step % logging_step == 0):
                val_stats, val_prediction = evaluate(model, val_loader)
                test_stats, test_prediction = evaluate(model, test_loader) if not config.only_dev else (val_stats, val_prediction)

                save_evidence = []
                for metric_name in metrics:
                    assert metric_name in val_stats.keys(), "Metrics not exist..."
                    score = best_scores[metric_name]
                    current_score = float(val_stats[metric_name])
                    if score is None or current_score > score:
                        save_evidence.append(metric_name)
                        best_scores[metric_name] = current_score

                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             'step': i+1,
                             'total_step': total_step
                             }
                
                need_tb_logs.update({
                    **{f'val/{k}': v for k, v in val_stats.items()},
                    **{f'test/{k}': v for k, v in test_stats.items()}
                })

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'step': i+1,
                    'total_step': total_step
                }

                if utils.is_main_process():
                    ckpt_sub_path = os.path.join(ckpt_output_path, f"epoch_{epoch}-step_{i}")
                    
                    # logging statements
                    utils.write_json(ckpt_sub_path, "log_stats", log_stats)
                    
                    # logging prediction
                    utils.write_json(ckpt_sub_path, "val_prediction", val_prediction)
                    utils.write_json(ckpt_sub_path, "test_prediction", test_prediction)

                    # Saving normal checkpoints
                    if save_evidence or config.save_every_checkpoint:
                        torch.save(save_obj, os.path.join(ckpt_sub_path, 'checkpoint.pth'))

                    # Saving checkpoints if they are distinct
                    for metric_name in save_evidence:
                        best_ckpt_path = os.path.join(ckpt_output_path, f"best_{metric_name}")
                        utils.copy_whole_dir(ckpt_sub_path, best_ckpt_path)
            
            tb_writer.add_dict_scalar(need_tb_logs, total_step)

        lr_scheduler.step(epoch+warmup_steps+1)
        if utils.is_dist_avail_and_initialized():
            dist.barrier()
        torch.cuda.empty_cache()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logger.info("Averaged stats: {}".format(metric_logger.summary(mode="avg")))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('***** Stopping training *****')
    logger.info('Training time {}'.format(total_time_str))
    tb_writer.close()


@torch.no_grad()
def evaluate(model, data_loader, special_name="Val"):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(logging=logger.info, delimiter=" - ")
    header = 'Evaluation: ' + special_name
    print_freq = 20
    predictions = []
    losses = []
    goldens = []
    orders = []
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch = utils.prepare_input(batch, device)
        outputs = model(batch)
        loss = outputs['loss']
        prediction = outputs['prediction']
        
        prediction = utils.concat_all_gather(prediction).to('cpu')
        loss = utils.concat_all_gather(loss.unsqueeze(0)).to('cpu')
        targets = utils.concat_all_gather(batch.label).to('cpu')
        order = utils.concat_all_gather(batch.order).to('cpu')
        
        predictions.append(prediction)
        losses.append(loss) 
        goldens.append(targets)
        orders.append(order)

    predictions = torch.cat(predictions)
    losses = torch.cat(losses)
    goldens = torch.cat(goldens)
    orders = torch.cat(orders)
    _, pred_class = predictions.max(1)

    accuracy = accuracy_score(pred_class, goldens)
    precision = precision_score(goldens, pred_class, average='macro')
    recall = recall_score(goldens, pred_class, average='macro')
    F1 = f1_score(goldens, pred_class, average='macro')

    eval_result = {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'F-1': F1.item(),
        'loss': losses.mean().item()
    }

    for metric, res in eval_result.items():
        logger.info(special_name + " Averaged {} is {}.".format(metric, res))

    id2data = data_loader.dataset.id2data
    predict_result = []
    for order, pred in zip(orders.tolist(), pred_class.tolist()):
        item = id2data[order]
        item['prediction'] = pred
        predict_result.append(item)

    return eval_result, predict_result


def data_prepare(tokenizer):

    logger.info("- - - - - - - - - - - - - Creating dataset- - - - - - - - - - - - - ")
    train_dataset, val_dataset, test_dataset = create_dataset('normal', config, tokenizer)
    if config.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(
            [train_dataset, val_dataset, test_dataset], [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4, 4, 4], is_trains=[True, False, False],
                                                          collate_fns=[train_dataset.collate_fn, val_dataset.collate_fn, test_dataset.collate_fn])
    return train_loader, val_loader, test_loader


def model_prepare():

    logger.info("- - - - - - - - - - - - - Creating model- - - - - - - - - - - - - ")
    model = BaselineBert(config=config, text_encoder=config.bert_config)
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info('load checkpoint from %s' % config.checkpoint)
        logger.info(msg)

    model = model.to(device)
    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.gpu], broadcast_buffers=False)
        model_without_ddp = model.module

    return model, model_without_ddp


def main():

    tokenizer = BertTokenizer.from_pretrained(config.bert_config)
    
    #### Dataset ####
    train_loader, val_loader, test_loader = data_prepare(tokenizer)

    #### Model ####
    model, model_without_ddp = model_prepare()

    #### Training Controler ####
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if config.eval_before_train:
        evaluate(model, val_loader)

    training_loop(model, model_without_ddp, train_loader,
                  val_loader, test_loader, optimizer,
                  tokenizer, lr_scheduler)


def parse_args():
    # See: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    parser = argparse.ArgumentParser(
        description="necessarily parameters for run this code."
    )
    parser.add_argument('--config', default='configs/baseline-bert.yaml')
    parser.add_argument('--output_dir', default='output/debug')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--eval_before_train', action='store_true')
    parser.add_argument('--only_dev', action='store_false')
    parser.add_argument('--dist_backend', default='nccl')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='gradient accumulation for increase batch virtually.')
    parser.add_argument('--max_grad_norm', default=5.0, type=float,
                        help='clip gradient norm of an iterable of parameters')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='device number of current process.') 
    parser.add_argument('--logging_step', default=500, type=int) 
    parser.add_argument('--logging_strategy', type=str, choices=['no','epoch','steps'], default='steps')
    parser.add_argument('--logging_level', type=str, choices=['DEBUG','INFO','ERROR','WARNING'], default='DEBUG')
    parser.add_argument('--save_every_checkpoint', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # set configuration for training or evaluating
    args = parse_args()
    config = utils.read_yaml(args.config)
    config = utils.AttrDict(config)
    args = utils.AttrDict(args.__dict__)
    # The parameters passed in from the command line take precedence
    config.update(args)

    # Determine global parameters and settings
    utils.init_distributed_mode(config)
    device = torch.device(config.device)
    # fix the seed for reproducibility
    utils.setup_seed(config.seed)
    # record them in file.
    logger, tb_writer = utils.create_logger(config)

    logger.debug(f"all global configuration is here: {str(config)}")

    main()
