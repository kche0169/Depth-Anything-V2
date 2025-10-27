import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# 导入多模态数据集类
from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from dataset.radarrgbd import RadarRGBD
# 导入原始和多模态模型
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log


'''
Example usage for multi-modal training:
python train.py --use-multimodal --use-depth-input --use-radar-input --pretrained-from path/to/checkpoint.pth --save-path ./multimodal_output --freeze-backbone --freeze-epochs 5
'''

parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='hypersim', choices=['hypersim', 'vkitti', 'radarrgbd'])
parser.add_argument('--data-path', type=str, help='path to dataset')
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

# 添加多模态相关的命令行参数
parser.add_argument('--use-multimodal', action='store_true', help='Use multi-modal model')
parser.add_argument('--use-depth-input', action='store_true', help='Use depth input as a feature')
parser.add_argument('--use-radar-input', action='store_true', help='Use radar input as a feature')
parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone weights during training')
parser.add_argument('--freeze-epochs', type=int, default=5, help='Number of epochs to freeze backbone')


def main():
    args = parser.parse_args()
    
    warnings.simplefilter('ignore', np.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    rank, world_size = setup_distributed(port=args.port)
    
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (args.img_size, args.img_size)
    
    # 根据参数选择合适的数据集类
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
    elif args.dataset == 'vkitti':
        trainset = VKITTI2('dataset/splits/vkitti2/train.txt', 'train', size=size)
    elif args.dataset == 'radarrgbd':
        trainset = RadarRGBD(args.data_path, 'dataset/splits/train.txt', 'train', size=size)
    else:
        raise NotImplementedError
        
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)
    
    # 验证集也使用多模态版本
    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    elif args.dataset == 'vkitti':
        valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    elif args.dataset == 'radarrgbd':
        valset = RadarRGBD(args.data_path, 'dataset/splits/val.txt', 'val', size=size)
    else:
        raise NotImplementedError
        
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # 根据参数选择模型
    if args.use_multimodal:
        logger.info(f"Creating multi-modal model with depth_input={args.use_depth_input}, radar_input={args.use_radar_input}")
        model = MultiModalDepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    else:
        model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    # 加载预训练权重
    if args.pretrained_from:
        if args.use_multimodal:
            # 为多模态模型加载权重
            checkpoint = torch.load(args.pretrained_from, map_location='cpu')
            
            # 仅加载RGB编码器权重
            if 'model' in checkpoint:
                checkpoint_weights = checkpoint['model']
            else:
                checkpoint_weights = checkpoint
                
            # 筛选出RGB编码器权重
            rgb_encoder_weights = {}
            for k, v in checkpoint_weights.items():
                # 将预训练权重映射到多模态模型中的rgb_encoder
                if 'pretrained' in k or 'depth_head' in k:
                    if args.use_multimodal:
                        # 将权重映射到新的模型结构
                        new_key = k.replace('pretrained', 'rgb_encoder')
                        rgb_encoder_weights[new_key] = v
                    else:
                        rgb_encoder_weights[k] = v
            
            # 加载RGB编码器权重
            model.load_state_dict(rgb_encoder_weights, strict=False)
            logger.info(f"Loaded RGB encoder weights from {args.pretrained_from}")
        else:
            model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                     output_device=local_rank, find_unused_parameters=True)
    
    criterion = SiLogLoss().cuda(local_rank)
    
    # 针对多模态模型调整优化器参数分组
    if args.use_multimodal:
        # 为不同部分设置不同的学习率
        param_groups = [
            # RGB编码器 - 较低学习率
            {'params': [param for name, param in model.named_parameters() 
                        if 'rgb_encoder' in name], 'lr': args.lr},
            # 深度编码器 - 较高学习率
            {'params': [param for name, param in model.named_parameters() 
                        if 'depth_encoder' in name], 'lr': args.lr * 10.0},
            # 雷达编码器 - 较高学习率
            {'params': [param for name, param in model.named_parameters() 
                        if 'radar_encoder' in name], 'lr': args.lr * 10.0},
            # 融合模块 - 较高学习率
            {'params': [param for name, param in model.named_parameters() 
                        if 'modal_fusion' in name], 'lr': args.lr * 10.0},
            # 深度头 - 较高学习率
            {'params': [param for name, param in model.named_parameters() 
                        if 'depth_head' in name], 'lr': args.lr * 10.0},
        ]
    else:
        param_groups = [
            {'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
            {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}
        ]
    
    optimizer = AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    total_iters = args.epochs * len(trainloader)
    
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    
    for epoch in range(args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
            logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                        'log10: {:.3f}, silog: {:.3f}'.format(
                            epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'], 
                            previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))
        
        # 根据参数冻结主干网络
        if args.freeze_backbone and epoch < args.freeze_epochs:
            if args.use_multimodal:
                for name, param in model.named_parameters():
                    if 'rgb_encoder' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                if rank == 0:
                    logger.info('Freezing RGB encoder backbone for epoch {}'.format(epoch))
            else:
                for name, param in model.named_parameters():
                    if 'pretrained' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                if rank == 0:
                    logger.info('Freezing backbone for epoch {}'.format(epoch))
        else:
            # 解冻所有参数
            for param in model.parameters():
                param.requires_grad = True
            if rank == 0 and epoch == args.freeze_epochs and args.freeze_backbone:
                logger.info('Unfreezing all parameters at epoch {}'.format(epoch))
        
        trainloader.sampler.set_epoch(epoch + 1)
        
        model.train()
        total_loss = 0
        
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            
            # 获取基本输入数据
            img = sample['image'].cuda()
            depth = sample['depth'].cuda() 
            valid_mask = sample['valid_mask'].cuda()
            
            # 获取多模态输入（如果有）
            depth_input = sample.get('depth_input')
            radar_input = sample.get('radar_input')
            
            if depth_input is not None:
                depth_input = depth_input.cuda()
            
            if radar_input is not None:
                radar_input = radar_input.cuda()
            
            # 随机水平翻转数据增强
            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)
                
                if depth_input is not None:
                    depth_input = depth_input.flip(-1)
                
                if radar_input is not None:
                    radar_input = radar_input.flip(-1)
            
            # 根据模型类型调用前向传播
            if args.use_multimodal:
                pred = model(img, depth_input, radar_input)
            else:
                pred = model(img)
            
            loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            
            # 更新学习率
            for idx, param_group in enumerate(optimizer.param_groups):
                if idx == 0:  # RGB编码器或预训练部分
                    param_group["lr"] = lr
                else:  # 其他部分使用更高学习率
                    param_group["lr"] = lr * 10.0
            
            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)
            
            if rank == 0 and i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))
        
        model.eval()
        
        results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
                   'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
        nsamples = torch.tensor([0.0]).cuda()
        
        for i, sample in enumerate(valloader):
            # 获取基本输入数据
            img = sample['image'].cuda().float()
            depth = sample['depth'].cuda()[0]
            valid_mask = sample['valid_mask'].cuda()[0]
            
            # 获取多模态输入（如果有）
            depth_input = sample.get('depth_input')
            radar_input = sample.get('radar_input')
            
            if depth_input is not None:
                depth_input = depth_input.cuda()
            
            if radar_input is not None:
                radar_input = radar_input.cuda()
            
            with torch.no_grad():
                # 根据模型类型调用前向传播
                if args.use_multimodal:
                    pred = model(img, depth_input, radar_input)
                else:
                    pred = model(img)
                    
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
            valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            
            if valid_mask.sum() < 10:
                continue
            
            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            
            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1
        
        torch.distributed.barrier()
        
        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)
        
        if rank == 0:
            logger.info('==========================================================================================')
            logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
            logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
            logger.info('==========================================================================================')
            print()
            
            for name, metric in results.items():
                writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
        
        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'args': vars(args),  # 保存配置
            }
            torch.save(checkpoint, os.path.join(args.save_path, f'checkpoint_epoch{epoch}.pth'))
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            
            # 如果是最佳结果，额外保存
            if previous_best['d1'] == (results['d1'] / nsamples).item() or previous_best['rmse'] == (results['rmse'] / nsamples).item():
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()