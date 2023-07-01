## Reference
LLFF的`imgs2poses.py`

https://github.com/Fyusion/LLFF/blob/master/imgs2poses.py

## Project

当colmap已经生成好`sparse/0`中的`cameras.bin, images.bin, points3D.bin`后，来根据其生成`poses_bounds.npy`（这个和输入的`factor`没有关系）
    
`poses_bounds.npy`：
- Each row of length 17 gets reshaped into a 3x5 **pose matrix** and 2 **depth values** that bound the closest and farthest scene content from that point of view（是离相机远近）. 
- pose matrix的方向是LLFF格式的DRB方向。

## Modification
- 已植入`minify_uniform`, 跨平台，这个才用`factor`来生成下采样的图片

## Run

`$dataset`是图片的父目录
```bash
python utils/imgs2poses/imgs2poses.py -d ~/dataset/Ollie -f 4
```