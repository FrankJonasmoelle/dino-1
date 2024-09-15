from main_dino import DataAugmentationDINO
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import utils
import json
import os
import time
import datetime
from pathlib import Path
import math
import sys
import config

from main_dino import DINOLoss
from utils import MultiCropWrapper
from vision_transformer import DINOHead, vit_small, vit_tiny, vit_base
import vision_transformer


class Dino():
    def __init__(self, architecture="vit_tiny"):
        self.student = vision_transformer.__dict__[architecture](config.patch_size) #, drop_path_rate)
        self.teacher = vision_transformer.__dict__[architecture](config.patch_size)

        embed_dim = self.student.embed_dim
        self.student = MultiCropWrapper(self.student, DINOHead(embed_dim, config.out_dim, config.use_bn_in_head, config.norm_last_layer))
        self.teacher = MultiCropWrapper(self.teacher, DINOHead(embed_dim, config.out_dim, config.use_bn_in_head))
                                    
        # move networks to gpu
        if torch.cuda.is_available():
            self.student, self.teacher = self.student.cuda(), self.teacher.cuda()
        print(f"using cuda: {torch.cuda.is_available()}")

        self.teacher_without_ddp = self.teacher
        # teacher and student start with the same weights
        self.teacher_without_ddp.load_state_dict(self.student.state_dict())
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        print(f"Student and Teacher are built: they are both {architecture} network.")

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, config.epochs)
    
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        if torch.cuda.is_available():
            images = [im.cuda(non_blocking=True) for im in images]
        else:
            print("cuda not available")
        
        # teacher and student forward passes + compute dino loss
        teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output = student(images)
        loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        loss.backward()
        if config.clip_grad:
            param_norms = utils.clip_gradients(student, config.clip_grad)
        utils.cancel_gradients_last_layer(epoch, student, config.freeze_last_layer)
        optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        # torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    
    # Print averaged stats
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    transform = DataAugmentationDINO(
        config.global_crops_scale,
        config.local_crops_scale,
        config.local_crops_number
    )

    dataset = datasets.ImageFolder(config.datapath, transform=transform)
    # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        # sampler=sampler,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")


    dino = Dino("vit_tiny")
    
    # loss
    if torch.cuda.is_available():
        dino_loss = DINOLoss(
            config.out_dim,
            config.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            config.warmup_teacher_temp,
            config.teacher_temp,
            config.warmup_teacher_temp_epochs,
            config.epochs,
        ).cuda()

    # optimizer
    params_groups = utils.get_params_groups(dino.student)
    optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs

    # schedulers
    lr_schedule = utils.cosine_scheduler(
        config.lr * (config.batch_size_per_gpu) / 256., # TODO CHeck!!!  * utils.get_world_size()  # linear scaling rule
        config.min_lr,
        config.epochs, len(data_loader),
        warmup_epochs=config.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        config.weight_decay,
        config.weight_decay_end,
        config.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(config.momentum_teacher, 1,
                                                config.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")


    print("Starting DINO training !")

    start_time = time.time()

    for epoch in range(0, config.epochs):
        # Training one epoch of DINO
        train_stats = train_one_epoch(
            dino.student, dino.teacher, dino.teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch
        )

        # Writing logs
        save_dict = {
            'student': dino.student.state_dict(),
            'teacher': dino.teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'dino_loss': dino_loss.state_dict(),
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(config.output_dir, 'checkpoint.pth')
        torch.save(save_dict, checkpoint_path)
        
        if config.saveckp_freq and epoch % config.saveckp_freq == 0:
            checkpoint_path = os.path.join(config.output_dir, f'checkpoint{epoch:04}.pth')
            torch.save(save_dict, checkpoint_path)
        
        # Logging statistics
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        
        # Always perform logging
        log_path = Path(config.output_dir) / "log.txt"
        with log_path.open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

    # Print total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))