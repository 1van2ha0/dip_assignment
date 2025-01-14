pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu117.html
python run.py --config configs/llff_3v/room.py --render_test_get_metric


(/home/mugi/VSCODE/PYTHON/VGOS/.conda) (.conda) (base) mugi@mugi:~/VSCODE/PYTHON/VGOS$ pip install numpy==1.26.4
Collecting numpy==1.26.4
  Using cached numpy-1.26.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
Using cached numpy-1.26.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
Installing collected packages: numpy
  Attempting uninstall: numpy
    Found existing installation: numpy 2.0.1
    Uninstalling numpy-2.0.1:
      Successfully uninstalled numpy-2.0.1
Successfully installed numpy-1.26.4
(/home/mugi/VSCODE/PYTHON/VGOS/.conda) (.conda) (base) mugi@mugi:~/VSCODE/PYTHON/VGOS$ python run.py --config configs/llff_3v/room.py --render_test_get_metric
Using /home/mugi/.cache/torch_extensions/py39_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/mugi/.cache/torch_extensions/py39_cu117/adam_upd_cuda/build.ninja...
Building extension module adam_upd_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module adam_upd_cuda...
Using /home/mugi/.cache/torch_extensions/py39_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/mugi/.cache/torch_extensions/py39_cu117/render_utils_cuda/build.ninja...
Building extension module render_utils_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module render_utils_cuda...
Using /home/mugi/.cache/torch_extensions/py39_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/mugi/.cache/torch_extensions/py39_cu117/total_variation_cuda/build.ninja...
Building extension module total_variation_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module total_variation_cuda...
Using /home/mugi/.cache/torch_extensions/py39_cu117 as PyTorch extensions root...
No modifications detected for re-loaded extension module render_utils_cuda, skipping build step...
Loading extension module render_utils_cuda...
Using /home/mugi/.cache/torch_extensions/py39_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/mugi/.cache/torch_extensions/py39_cu117/color_aware_voxel_smooth_cuda/build.ninja...
Building extension module color_aware_voxel_smooth_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module color_aware_voxel_smooth_cuda...
Loading images from ./data/nerf_llff_data/room/images_1008x756
Loaded image data (3024, 4032, 3, 41) [3024.         4032.          767.65956772]
Loaded ./data/nerf_llff_data/room 10.706691140704915 91.66782174279389
recentered (3, 5)
[[ 1.0000000e+00 -3.2399280e-10 -7.2935608e-10  7.9957454e-09]
 [ 3.2399280e-10  1.0000000e+00  9.5727992e-10 -1.4537718e-09]
 [ 7.2935608e-10 -9.5727992e-10  1.0000000e+00  2.3169489e-09]]
Data:
(41, 3, 5) (41, 3024, 4032, 3) (41, 2)
HOLDOUT view is 35
Loaded llff (41, 3024, 4032, 3) torch.Size([120, 3, 5]) [3024.      4032.       767.65955] ./data/nerf_llff_data/room
Auto LLFF holdout, 8
DEFINING BOUNDS
NEAR FAR 0.0 1.0
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Currently logged in as: 3023625451 (3023625451-university-of-science-and-technology-of-china). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.1
wandb: Run data is saved locally in /home/mugi/VSCODE/PYTHON/VGOS/wandb/run-20241226_195709-np9aq8y0
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run dry-silence-2
wandb: ⭐️ View project at https://wandb.ai/3023625451-university-of-science-and-technology-of-china/SVGO
wandb: 🚀 View run at https://wandb.ai/3023625451-university-of-science-and-technology-of-china/SVGO/runs/np9aq8y0
wandb: WARNING Calling wandb.run.save without any arguments is deprecated.Changes to attributes are automatically persisted.
train: start
compute_bbox_by_cam_frustrm: start
/home/mugi/VSCODE/PYTHON/VGOS/.conda/lib/python3.9/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /opt/conda/conda-bld/pytorch_1666642975993/work/aten/src/ATen/native/TensorShape.cpp:3190.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
compute_bbox_by_cam_frustrm: xyz_min tensor([-1.7708, -1.7860, -1.0000])
compute_bbox_by_cam_frustrm: xyz_max tensor([1.6864, 1.6552, 1.0000])
compute_bbox_by_cam_frustrm: finish
train: skip coarse geometry searching
Original training views: [ 1  2  3  4  5  6  7  9 10 11 12 13 14 15 17 18 19 20 21 22 23 25 26 27
 28 29 30 31 33 34 35 36 37 38 39]
Subsampled train views: [18 10  4]
scene_rep_reconstruction (fine): train from scratch
scene_rep_reconstruction (fine): use multiplane images
dmpigo: world_size       tensor([ 90,  90, 128])
dmpigo: voxel_size_ratio 2.0
dmpigo: densitye grid DenseGrid(channels=1, world_size=[90, 90, 128])
dmpigo: feature grid DenseGrid(channels=9, world_size=[90, 90, 128])
dmpigo: mlp Sequential(
  (0): Linear(in_features=12, out_features=64, bias=True)
  (1): ReLU(inplace=True)
  (2): Sequential(
    (0): Linear(in_features=64, out_features=64, bias=True)
    (1): ReLU(inplace=True)
  )
  (3): Linear(in_features=64, out_features=3, bias=True)
)
create_optimizer_or_freeze_model: param density lr 0.1
create_optimizer_or_freeze_model: param k0 lr 0.1
create_optimizer_or_freeze_model: param rgbnet lr 0.001
get_training_rays: start
get_training_rays: finish (eps time: 0.5461463928222656 sec)
get_ramdom_rays: start
get_random_rays: finish (eps time: 0.013283967971801758 sec)
  6%|████████████▌                                                                                                                                                                                                                      | 497/9000 [00:15<04:35, 30.88it/s]dmpigo: update mask_cache 1.0000 => 1.0000
scene_rep_reconstruction (fine): iter   1000 / Loss: 0.003036285 / PSNR: 23.96 / Eps: 00:00:30                                                                                                                                                                             
 11%|█████████████████████████▏                                                                                                                                                                                                         | 997/9000 [00:30<03:45, 35.52it/sTesting (3024, 4032, 3)                                                                                                                                                                                                                               | 0/2 [00:00<?, ?it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:18<00:00,  9.39s/it]
 17%|█████████████████████████████████████▋                                                                                                                                                                                            | 1499/9000 [01:22<03:25, 36.43it/s]dmpigo: update mask_cache 1.0000 => 1.0000
 22%|██████████████████████████████████████████████████▏                                                                                                                                                                               | 1999/9000 [01:35<03:02, 38.40it/s]dmpigo: scale_volume_grid start
dmpigo: world_size       tensor([128, 127, 128])
dmpigo: voxel_size_ratio 2.0
dmpigo: scale_volume_grid scale world_size from [90, 90, 128] to [128, 127, 128]
dmpigo: scale_volume_grid finish
create_optimizer_or_freeze_model: param density lr 0.1
create_optimizer_or_freeze_model: param k0 lr 0.1
create_optimizer_or_freeze_model: param rgbnet lr 0.001
scene_rep_reconstruction (fine): iter   2000 / Loss: 0.008353594 / PSNR: 26.21 / Eps: 00:01:36                                                                                                                                                                             
 22%|██████████████████████████████████████████████████▏                                                                                                                                                                               | 1999/9000 [01:35<03:02, 38.40it/sTesting (3024, 4032, 3)                                                                                                                                                                                                                               | 0/2 [00:00<?, ?it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:13<00:00,  6.86s/it]
 22%|██████████████████████████████████████████████████▏                                                                                                                                                                               | 1999/9000 [01:50<03:02, 38.40it/sTesting (3024, 4032, 3)                                                                                                                                                                                                                               | 0/6 [00:00<?, ?it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [01:32<00:00, 15.34s/it]
 28%|██████████████████████████████████████████████████████████████▋                                                                                                                                                                   | 2495/9000 [03:51<02:40, 40.51it/s]dmpigo: update mask_cache 1.0000 => 1.0000
scene_rep_reconstruction (fine): iter   3000 / Loss: 0.002309951 / PSNR: 26.26 / Eps: 00:04:04                                                                                                                                                                             
 33%|███████████████████████████████████████████████████████████████████████████▏                                                                                                                                                      | 2996/9000 [04:03<02:29, 40.27it/sTesting (3024, 4032, 3)                                                                                                                                                                                                                               | 0/2 [00:00<?, ?it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:15<00:00,  7.53s/it]
 39%|███████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                          | 3495/9000 [04:49<02:15, 40.67it/s]dmpigo: update mask_cache 1.0000 => 1.0000
 44%|████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                             | 3995/9000 [05:02<02:03, 40.66it/s]dmpigo: scale_volume_grid start
dmpigo: world_size       tensor([181, 180, 128])
dmpigo: voxel_size_ratio 2.0
dmpigo: scale_volume_grid scale world_size from [128, 127, 128] to [181, 180, 128]
dmpigo: scale_volume_grid finish
create_optimizer_or_freeze_model: param density lr 0.1
create_optimizer_or_freeze_model: param k0 lr 0.1
create_optimizer_or_freeze_model: param rgbnet lr 0.001
scene_rep_reconstruction (fine): iter   4000 / Loss: 0.006658279 / PSNR: 27.10 / Eps: 00:05:02                                                                                                                                                                             
 44%|████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                             | 3995/9000 [05:02<02:03, 40.66it/sTesting (3024, 4032, 3)                                                                                                                                                                                                                               | 0/2 [00:00<?, ?it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:11<00:00,  5.79s/it]
 44%|████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                             | 3995/9000 [05:20<02:03, 40.66it/sTesting (3024, 4032, 3)                                                                                                                                                                                                                               | 0/6 [00:00<?, ?it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [01:27<00:00, 14.63s/it]
 50%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                 | 4499/9000 [07:09<01:37, 46.05it/s]dmpigo: update mask_cache 1.0000 => 1.0000
scene_rep_reconstruction (fine): iter   5000 / Loss: 0.002198908 / PSNR: 26.84 / Eps: 00:07:21                                                                                                                                                                             
 56%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                    | 4999/9000 [07:21<01:34, 42.56it/sTesting (3024, 4032, 3)                                                                                                                                                                                                                               | 0/2 [00:00<?, ?it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:13<00:00,  6.54s/it]
 61%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                        | 5499/9000 [08:04<01:21, 42.80it/s]dmpigo: update mask_cache 1.0000 => 1.0000
 67%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                           | 5999/9000 [08:16<01:12, 41.27it/s]dmpigo: scale_volume_grid start
dmpigo: world_size       tensor([256, 255, 128])
dmpigo: voxel_size_ratio 2.0
dmpigo: scale_volume_grid scale world_size from [181, 180, 128] to [256, 255, 128]
dmpigo: scale_volume_grid finish
create_optimizer_or_freeze_model: param density lr 0.1
create_optimizer_or_freeze_model: param k0 lr 0.1
create_optimizer_or_freeze_model: param rgbnet lr 0.001
scene_rep_reconstruction (fine): iter   6000 / Loss: 0.006351071 / PSNR: 27.56 / Eps: 00:08:17                                                                                                                                                                             
 67%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                           | 5999/9000 [08:16<01:12, 41.27it/sTesting (3024, 4032, 3)                                                                                                                                                                                                                               | 0/2 [00:00<?, ?it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:10<00:00,  5.48s/it]
 67%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                           | 5999/9000 [08:31<01:12, 41.27it/sTesting (3024, 4032, 3)                                                                                                                                                                                                                               | 0/6 [00:00<?, ?it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [01:25<00:00, 14.30s/it]
 72%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                               | 6495/9000 [10:21<01:00, 41.49it/s]dmpigo: update mask_cache 0.9476 => 0.9476
scene_rep_reconstruction (fine): iter   7000 / Loss: 0.002071811 / PSNR: 27.17 / Eps: 00:10:34                                                                                                                                                                             
 78%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                  | 6995/9000 [10:34<00:49, 40.63it/sTesting (3024, 4032, 3)                                                                                                                                                                                                                               | 0/2 [00:00<?, ?it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:12<00:00,  6.16s/it]
 83%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                     | 7497/9000 [11:18<00:39, 38.25it/s]dmpigo: update mask_cache 0.9476 => 0.9476
 89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                         | 7997/9000 [11:31<00:26, 37.43it/s]dmpigo: scale_volume_grid start
dmpigo: world_size       tensor([362, 361, 128])
dmpigo: voxel_size_ratio 2.0
dmpigo: scale_volume_grid scale world_size from [256, 255, 128] to [362, 361, 128]
dmpigo: scale_volume_grid finish
create_optimizer_or_freeze_model: param density lr 0.1
create_optimizer_or_freeze_model: param k0 lr 0.1
create_optimizer_or_freeze_model: param rgbnet lr 0.001
scene_rep_reconstruction (fine): iter   8000 / Loss: 0.006487479 / PSNR: 27.81 / Eps: 00:11:32                                                                                                                                                                             
 89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                         | 7997/9000 [11:31<00:26, 37.43it/sTesting (3024, 4032, 3)                                                                                                                                                                                                                               | 0/2 [00:00<?, ?it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:10<00:00,  5.38s/it]
 89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                         | 7997/9000 [11:51<00:26, 37.43it/sTesting (3024, 4032, 3)                                                                                                                                                                                                                               | 0/6 [00:00<?, ?it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [01:25<00:00, 14.19s/it]
 94%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎            | 8496/9000 [13:38<00:15, 33.08it/s]dmpigo: update mask_cache 0.5768 => 0.5768
scene_rep_reconstruction (fine): iter   9000 / Loss: 0.002538833 / PSNR: 27.19 / Eps: 00:13:54                                                                                                                                                                             
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉| 8996/9000 [13:54<00:00, 31.81it/sTesting (3024, 4032, 3)                                                                                                                                                                                                                               | 0/2 [00:00<?, ?it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:11<00:00,  5.87s/it]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9000/9000 [14:25<00:00, 10.40it/s]
scene_rep_reconstruction (fine): saved checkpoints at ./logs/llff_3v_cpr/room_3v/fine_last.tar
train: fine detail reconstruction in 00:14:34
train: finish (eps time 00:14:35 )
dmpigo: world_size       tensor([362, 361, 128])
dmpigo: voxel_size_ratio 2.0
dmpigo: densitye grid DenseGrid(channels=1, world_size=[362, 361, 128])
dmpigo: feature grid DenseGrid(channels=9, world_size=[362, 361, 128])
dmpigo: mlp Sequential(
  (0): Linear(in_features=12, out_features=64, bias=True)
  (1): ReLU(inplace=True)
  (2): Sequential(
    (0): Linear(in_features=64, out_features=64, bias=True)
    (1): ReLU(inplace=True)
  )
  (3): Linear(in_features=64, out_features=3, bias=True)
)
All results are dumped into ./logs/llff_3v_cpr/room_3v
  0%|                                                                                                                                                                                                                                                | 0/6 [00:00<?, ?it/s]Testing (3024, 4032, 3)
init_lpips: lpips_alex
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/home/mugi/VSCODE/PYTHON/VGOS/.conda/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/mugi/VSCODE/PYTHON/VGOS/.conda/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth" to /home/mugi/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 233M/233M [00:09<00:00, 24.9MB/s]
Loading model from: /home/mugi/VSCODE/PYTHON/VGOS/.conda/lib/python3.9/site-packages/lpips/weights/v0.1/alex.pth██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌ | 232M/233M [00:09<00:00, 23.9MB/s]
init_lpips: lpips_vgg
Setting up [LPIPS] perceptual loss: trunk [vgg], v[0.1], spatial [off]
/home/mugi/VSCODE/PYTHON/VGOS/.conda/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /home/mugi/.cache/torch/hub/checkpoints/vgg16-397923af.pth
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:35<00:00, 15.7MB/s]
Loading model from: /home/mugi/VSCODE/PYTHON/VGOS/.conda/lib/python3.9/site-packages/lpips/weights/v0.1/vgg.pth███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████  | 523M/528M [00:35<00:00, 20.5MB/s]
  0%|                                                                                                                                                                                                                                                | 0/6 [01:02<?, ?it/s]
Traceback (most recent call last):
  File "/home/mugi/VSCODE/PYTHON/VGOS/run.py", line 1478, in <module>
    rgbs, depths, bgmaps, psnrs, ssims, lpips_alex, lpips_vgg = render_viewpoints_and_metric(
  File "/home/mugi/VSCODE/PYTHON/VGOS/.conda/lib/python3.9/site-packages/torch/autograd/grad_mode.py", line 27, in decorate_context
    return func(*args, **kwargs)
  File "/home/mugi/VSCODE/PYTHON/VGOS/run.py", line 381, in render_viewpoints_and_metric
    lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))
  File "/home/mugi/VSCODE/PYTHON/VGOS/lib/utils.py", line 132, in rgb_lpips
    return __LPIPS__[net_name](gt, im, normalize=True).item()
  File "/home/mugi/VSCODE/PYTHON/VGOS/.conda/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1190, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/mugi/VSCODE/PYTHON/VGOS/.conda/lib/python3.9/site-packages/lpips/lpips.py", line 124, in forward
    diffs[kk] = (feats0[kk]-feats1[kk])**2
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.91 GiB (GPU 0; 23.62 GiB total capacity; 19.40 GiB already allocated; 340.56 MiB free; 22.32 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
Traceback (most recent call last):
  File "/home/mugi/VSCODE/PYTHON/VGOS/run.py", line 1478, in <module>
    rgbs, depths, bgmaps, psnrs, ssims, lpips_alex, lpips_vgg = render_viewpoints_and_metric(
  File "/home/mugi/VSCODE/PYTHON/VGOS/.conda/lib/python3.9/site-packages/torch/autograd/grad_mode.py", line 27, in decorate_context
    return func(*args, **kwargs)
  File "/home/mugi/VSCODE/PYTHON/VGOS/run.py", line 381, in render_viewpoints_and_metric
    lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))
  File "/home/mugi/VSCODE/PYTHON/VGOS/lib/utils.py", line 132, in rgb_lpips
    return __LPIPS__[net_name](gt, im, normalize=True).item()
  File "/home/mugi/VSCODE/PYTHON/VGOS/.conda/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1190, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/mugi/VSCODE/PYTHON/VGOS/.conda/lib/python3.9/site-packages/lpips/lpips.py", line 124, in forward
    diffs[kk] = (feats0[kk]-feats1[kk])**2
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.91 GiB (GPU 0; 23.62 GiB total capacity; 19.40 GiB already allocated; 340.56 MiB free; 22.32 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF


python run.py --config configs/llff_3v/fern.py --render_test_get_metric