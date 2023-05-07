import os
from subprocess import check_output
import numpy as np
from skimage import io
from skimage.transform import rescale, resize



def minify(basedir, factors=[], resolutions=[]):
    '''
    NOTE: 
    - run in linux, `cp` and `rm`. 
    - And a special cli `mogrify`
    
    according to the `factor` or `resolution` to scale the images in `images` and save the result to the corresponding subdirectory `images_{}` or `images_{}x{}`.
    `factor` and `resolution` can be both used.
    
    :param basedir: the parent directory of `images` directory
    :param factors: HW同比例缩放几倍, list for many different factors, e.g. `images_4` and `images_8` for the `factors=[4, 8]`
    :param resolutions: HW缩放到某个尺寸, e.g. `resolutions=[(300, 400)]`, 新图片的HW=(300,400)
    '''

    # 根据其要求的文件夹，如果有一个不存在，就重新生成。否则，都在就直接return。
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    
    # 原图片路径
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        # 判断其是factors还是resolutions
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(int(100./r))
        else:
            # mogrify 的 Geometry 是 WH
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])

        # 新生成的图片路径
        imgdir = os.path.join(basedir, name)
        # 跳过那些已经存在的
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)

        # 1. 创建新文件夹
        # 2. 先复制原图片到新文件夹         
        # 3. 跳转到新文件夹，使用mogrify命令缩放
        # 4. 跳转回去
        # 5. 删除复制的过来的原格式图片，我们要转成png，如果原格式ext是png，那么就不用删除（因为mogrify处理相同格式时，the original image file is overwritten）。
        os.makedirs(imgdir)

        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def minify_uniform(basedir, factors=[], resolutions=[]):
    '''
    NOTE: 
    - No special needs: suitable for all platforms, using `skimage` to downscale instead of using special cli `mogrify`
    
    according to the `factor` or `resolution` to scale the images in `images` and save the result to the corresponding subdirectory `images_{}` or `images_{}x{}`.
    `factor` and `resolution` can be both used.
    
    :param basedir: the parent directory of `images` directory
    :param factors: HW同比例缩放几倍, list for many different factors, e.g. `images_4` and `images_8` for the `factors=[4, 8]`
    :param resolutions: HW缩放到某个尺寸, e.g. `resolutions=[(300, 400)]`, 新图片的HW=(300,400)
    '''

    # 根据其要求的文件夹，如果有一个不存在，就重新生成。否则，都在就直接return。
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]

    # 供保存图片使用
    img_names = [f.split(os.path.sep)[-1] for f in imgs]
    # 读取图片
    imgs = np.stack([io.imread(img)/255. for img in imgs], 0)
    
    
    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
        else:
            # 和mogrify的WH显示结果一致
            name = 'images_{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)

        if os.path.exists(imgdir):
            continue
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        
        for i in range(imgs.shape[0]):
            if isinstance(r, int):
                imgs_down = rescale(imgs[i], 1.0/r, anti_aliasing=True, channel_axis=2)
            else:
                # 实际做的时候还是skimage的HW
                imgs_down = resize(imgs[i], (r[0], r[1]), anti_aliasing=True)
            imgs_down = (imgs_down * 255).astype(np.uint8)
            io.imsave(os.path.join(imgdir, img_names[i]), imgs_down)
            
if __name__ == '__main__':
    minify(r'C:\Users\lab\Pictures\NeuLF_dataset\Statue', factors=[4])
    minify_uniform(r'C:\Users\lab\Pictures\NeuLF_dataset\Statue', factors=[4])