# Tutorial for SUSTeck1K

---

## Download the SUSTeck1K dataset 

Download the dataset from the [link](https://github.com/ShiqiYu/OpenGait)

## Download the checkpoint 

1. Download whole_cat7-130000.pt from [link](http://www.cwang93.net:1000/share/YnZ8X9JF)
2. Put it to `output/Baseline/whole_cat7/checkpoints`

# Train/Test the dataset
```cd SUSTeck1K```
- Modify the `dataset_root` in `config/lidargait/lidargait_sustech1k.yaml`
- **Train**:
`CUDA_VISIBLE_DEVICES=xx python -m torch.distributed.launch --nproc_per_node=xx --master_port xx opengait/main.py --cfgs config/lidargait/lidargait_sustech1k.yaml --phase train`
- **Test**:
`CUDA_VISIBLE_DEVICES=xx python -m torch.distributed.launch --nproc_per_node=xx --master_port xx opengait/main.py --cfgs config/lidargait/lidargait_sustech1k.yaml --phase test`


# Dataset process

### Deformable scene flow generate
- ```cd datasets & python point2depth.py```