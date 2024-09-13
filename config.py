
datapath = "/Users/jonas/Desktop/dino/imagenet/ILSVRC/Data/CLS-LOC/train" # path to train

global_crops_scale = (0.14, 1.0) #Scale range of the cropped image before resizing, relatively to the origin image.
                                 #Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
                                 #recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)
local_crops_number = 8 # Number of small
                       # local views to generate. Set this parameter to 0 to disable multi-crop training.
                       # When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1.
local_crops_scale = (0.05, 0.4) # Scale range of the cropped image before resizing, relatively to the origin image.
                                 # Used for small local view cropping of multi-crop.
batch_size = 64 # batch-size : number of distinct images loaded on one GPU.'
num_workers = 10 # number of data loading workers


# DINO
out_dim = 65536 # "Dimensionality of the DINO head output. For complex and large datasets large values (like 65k) work well.""")
# embed_dim = 768 # defined in vision transformer 
use_bn_in_head = False # Whether to use batch normalizations in projection head (Default: False)
arch = "tiny" # Name of architecture to train. For quick experiments with ViTs, we recommend using vit_tiny or vit_small.
embed_dim = 192 # embed dim for vit -> differs for vit-tiny, vit-small, vit-base
patch_size = 16 # Size in pixels of input square patches - default 16 (for 16x16 patches). 
                # Using smaller values leads to better performance but requires more memory. 
                # Applies only for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
                # mixed precision training (--use_fp16 false) to avoid unstabilities.
drop_path_rate = 0.1 # stochastic depth rate
norm_last_layer = False # Whether or not to weight normalize the last layer of the DINO head.
                       # Not normalizing leads to better performance but can make the training unstable.
                       # In our experiments, we typically set this paramater to False with vit_small and True with vit_base.

# training
warmup_teacher_temp = 0.04 # Initial value for the teacher temperature: 0.04 works well in most cases.
                           # Try decreasing it if the training loss does not decrease.
teacher_temp = 0.04        # Final value (after linear warmup)
                           # of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
                           # starting with the default value of 0.04 and increase this slightly if needed.
warmup_teacher_temp_epochs = 0 # Number of warmup epochs for the teacher temperature (Default: 30).
epochs = 100
freeze_last_layer = 1 # Number of epochs
                      # during which we keep the output layer fixed. Typically doing so during
                      # the first epoch helps training. Try increasing this value if the loss does not decrease.

# optimizer and schedulers
lr = 0.0005 # Learning rate at the end of
            # linear warmup (highest LR used during training). The learning rate is linearly scaled
            # with the batch size, and specified here for a reference batch size of 256.
min_lr = 1e-6 # Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.
warmup_epochs = 10 # Number of epochs for the linear learning-rate warm up.
batch_size_per_gpu = 64 # Per-GPU batch-size : number of distinct images loaded on one GPU.'
weight_decay = 0.04 # Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.
weight_decay_end = 0.4 # Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by
                       # the end of training improves performance for ViTs.
momentum_teacher = 0.996 # Base EMA parameter for teacher update. 
                         # The value is increased to 1 during training with cosine schedule.
                         # We recommend setting a higher value with small batches: 
                         # for example use 0.9995 with batch size of 256."

output_dir = "./"
saveckp_freq = 20 # Save checkpoint every x epochs.
clip_grad = 3.0 # Maximal parameter gradient norm if using gradient clipping. 
                # Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 
                # 0 for disabling.