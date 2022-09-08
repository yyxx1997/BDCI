import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import torch
import torch.backends.cudnn as cudnn
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



def train_net(model, model_without_ddp, train_loader, val_loader, test_loader, optimizer, tokenizer, device, lr_scheduler, config):

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    logging_step = config['logging_step']
    output_dir = config['output_dir']
    metrics = config['metrics']
    r_drop_rate = config['r_drop']
    best_scores = {}
    for metric in metrics:
        best_scores[metric] = -9999
    best_step = 0
    total_step = 0
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size
    K = config.gradient_accumulation_steps
    my_context = model.no_sync if config.local_rank != -1 and i % K != 0 else nullcontext
    total_train_batch_size = config['batch_size_train'] * K * config.world_size

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    print("Start training")
    print("***** Running training *****")
    print(f"  Num examples = {len(train_loader.dataset)}")
    print(f"  Num Epochs = {config.schedular['epochs']}")
    print(f"  Instantaneous batch size per device = {config['batch_size_train']}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    print(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    start_time = time.time()

    for epoch in range(0, max_epoch):
        
        header = 'Train Epoch: [{}]'.format(epoch)
        if config.distributed:
            train_loader.sampler.set_epoch(epoch)

        for i, (caption, order, targets) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            model.train()
            targets = targets.to(device, non_blocking=True)
            text_inputs = tokenizer(caption, padding='longest', return_tensors="pt").to(device)

            with my_context():
                output = model(text_inputs, targets=targets, train=True)
                loss = output['loss']
                prediction = output['prediction']
                if r_drop_rate:
                    output_hat = model(text_inputs, targets=targets, train=True)
                    loss_hat = output_hat['loss']
                    prediction_hat = output_hat['prediction']
                    ce_loss = 0.5 * (loss + loss_hat)
                    kl_loss = compute_kl_loss(prediction, prediction_hat)
                    loss = ce_loss + r_drop_rate * kl_loss
                loss = loss / K
                loss.backward()
            if (i+1) % K == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_step += 1
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(loss=loss.item())

            if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
                lr_scheduler.step(i//step_size)
            
            if total_step % logging_step == 0:
                val_stats, _ = evaluate(model, val_loader, tokenizer, device, config)
                test_stats, test_prediction = evaluate(model, test_loader,tokenizer, device, config)

                if utils.is_main_process():
                    # logging statements
                    log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                                 **{f'test_{k}': v for k, v in test_stats.items()},
                                 'epoch': epoch,
                                 'step': total_step,
                                 'best_scores': best_scores,
                                 'best_step': best_step
                                 }
                    with open(os.path.join(output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    # Saving checkpoints
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'step': total_step
                    }
                    if config.save_every_checkpoint:
                        torch.save(save_obj, os.path.join(output_dir, 'checkpoint-step-{}.pth'.format(total_step)))
                    for metric_name, score in best_scores.items():
                        assert metric_name in val_stats.keys(), "Metrics not exist..."
                        current_score = val_stats[metric_name]
                        if float(current_score) > score:
                            torch.save(save_obj, os.path.join(output_dir, f'checkpoint_{metric_name}_best.pth'))
                            with open(os.path.join(output_dir, f"testset_{metric_name}_best.json"), "w") as f:
                                f.write(json.dumps(test_prediction, ensure_ascii=False, indent=4))
                            best_scores[metric_name] = float(current_score)
                            best_step = total_step

        lr_scheduler.step(epoch+warmup_steps+1)
        if utils.is_dist_avail_and_initialized():
            dist.barrier()
        torch.cuda.empty_cache()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())
        train_stats = {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
        val_stats, _ = evaluate(model, val_loader, tokenizer, device, config)
        test_stats, test_prediction = evaluate(model, test_loader, tokenizer, device, config)

        if utils.is_main_process():

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'best_scores': best_scores,
                         }
            with open(os.path.join(output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
                'step': total_step
            }
            if config.save_every_epoch:
                torch.save(save_obj, os.path.join(output_dir, 'checkpoint-epoch-{}.pth'.format(epoch)))
            for metric_name, score in best_scores.items():
                assert metric_name in val_stats.keys(), "Metrics not exist..."
                current_score = val_stats[metric_name]
                if float(current_score) > score:
                    torch.save(save_obj, os.path.join(output_dir, f'checkpoint_{metric_name}_best.pth'))
                    with open(os.path.join(output_dir, f"testset_{metric_name}_best.json"), "w") as f:
                        f.write(json.dumps(test_prediction, ensure_ascii=False, indent=4))
                    best_scores[metric_name] = float(current_score)
                    best_step = total_step
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    training_statis = {
        "best_step":best_step,
        "best_scores":best_scores
    }
    return training_statis


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    predictions = []
    goldens = []
    orders = []
    for i, (caption, order, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        order ,targets = order.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        text_inputs = tokenizer(caption, padding='longest', return_tensors="pt").to(device)
        prediction = model(text_inputs, train=False)['prediction']
        prediction = utils.concat_all_gather(prediction, config['dist']).to('cpu')
        targets = utils.concat_all_gather(targets, config['dist']).to('cpu')
        order = utils.concat_all_gather(order, config['dist']).to('cpu')
        predictions.append(prediction)
        goldens.append(targets)
        orders.append(order)

    predictions = torch.cat(predictions)
    goldens = torch.cat(goldens)
    orders = torch.cat(orders)
    _, pred_class = predictions.max(1)

    accuracy = accuracy_score(pred_class,goldens)
    precision = precision_score(goldens,pred_class,average='macro')
    recall = recall_score(goldens,pred_class,average='macro')
    F1 = f1_score(goldens,pred_class,average='macro')

    print("evaluation dataset size is ", goldens.size(0))
    print("Averaged stats accuracy:", accuracy)
    print("Averaged stats precision:", precision)
    print("Averaged stats recall:", recall)
    print("Averaged stats F1:", F1)
    eval_result = {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'F-1': F1.item()
    }
    
    id2data = data_loader.dataset.id2data
    predict_result = []
    for order, pred in zip(orders.tolist(), pred_class.tolist()):
        item = id2data[order]
        item['prediction'] = pred
        predict_result.append(item)

    return eval_result, predict_result


def data_prepare(config):

    print("\n-------------\nCreating dataset\n-------------\n")
    datasets = create_dataset('normal', config)
    if config.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(
            datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets, samplers,
                                                          batch_size=[
                                                              config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4, 4, 4], is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])
    return train_loader, val_loader, test_loader


def model_prepare(config, device):

    print("\n-------------\nCreating model\n-------------\n")
    model = BaselineBert(config=config, text_encoder=config.bert_config)

    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % config.checkpoint)
        print(msg)

    model = model.to(device)

    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.gpu], broadcast_buffers=False)
        model_without_ddp = model.module

    return model, model_without_ddp


def main(config):
    utils.init_distributed_mode(config)
    config['dist'] = config.distributed
    device = torch.device(config.device)

    # fix the seed for reproducibility
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    train_loader, val_loader, test_loader = data_prepare(config)

    #### Model ####
    tokenizer = BertTokenizer.from_pretrained(config.bert_config)
    model, model_without_ddp = model_prepare(config, device)

    #### Training Controler ####
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if config.eval_before_train:
        val_stats, _ = evaluate(model, val_loader, tokenizer, device, config)
        test_stats, _ = evaluate(model, test_loader, tokenizer, device, config)
    training_statis = train_net(model, model_without_ddp, train_loader, val_loader, test_loader, optimizer, tokenizer, device, lr_scheduler, config)

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write("training statistic infomation is: {}".format(training_statis))


def parse_args():
    parser = argparse.ArgumentParser(
        description="necessarily parameters for run this code."
    )
    parser.add_argument('--config', default='configs/baseline-bert.yaml')
    parser.add_argument('--output_dir', default='output/debug')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--eval_before_train', action='store_true')
    parser.add_argument('--dist_backend', default='nccl')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='gradient accumulation for increase batch virtually.')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='device number of current process.') 
    parser.add_argument('--logging_step', default=1000, type=int) 
    parser.add_argument('--save_every_checkpoint', default=False, type=bool)
    parser.add_argument('--save_every_epoch', default=False, type=bool)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # set configuration for training or evaluating
    args = parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config = utils.AttrDict(config)
    args = utils.AttrDict(args.__dict__)
    config.update(args)

    print("all global configuration is here:\n", config)
    if utils.is_main_process():
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        yaml.dump(dict(config), open(os.path.join(
            config.output_dir, 'global_config.yaml'), 'w'))
    main(config)
