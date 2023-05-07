## Reference
LLFF的`imgs2poses.py`

https://github.com/Fyusion/LLFF/blob/master/imgs2poses.py

## Project

当colmap已经生成好`sparse/0`中的`cameras.bin, images.bin, points3D.bin`后，来根据其生成`poses_bounds.npy`

- 已植入`minify_uniform`

## Run

`$dataset`是图片的父目录
```bash
# python utils/imgs2poses/imgs2poses.py $dataset
python utils/imgs2poses/imgs2poses.py ~/dataset/Ollie --factor 4
```