import matplotlib as mpl
import numpy as np


def colored_depthmap(
    depth_target: np.ndarray, 
    d_min: int = None, 
    d_max: int = None
) -> np.ndarray:
    '''
    @param depth_target: ndarry, uint8 or float64, [H, W] or [H, W, 1]
    @return: uint8, [H, W, 3]
    '''
    depth_target = np.squeeze(depth_target)
    if d_min is None:
        d_min = np.min(depth_target)
    if d_max is None:
        d_max = np.max(depth_target)
    # normalize
    depth_target = (depth_target - d_min) / (d_max - d_min)
    # colormap
    cmap = mpl.colormaps['viridis']
    # colormap and drop Alpha channel generated by cmap()
    depth_target = cmap(depth_target)[:,:,:3]
    # uint8
    depth_target = (depth_target * 255).astype("uint8")
    return depth_target


def main():

    from PIL import Image
    import matplotlib.pyplot as plt

    def merge_into_row(input, depth_target):
        '''
        @param input: 原始图片
        @param depth_target: 深度图片

        note: input and depth_target 的格式是PIL还是ndarry都行，用什么库打开都行，在函数中都会被处理为ndarry。
        '''
        input = np.jaxtyping.numpy.ndarray(input)
        depth_target = np.jaxtyping.numpy.ndarray(depth_target)
        depth_target_col = colored_depthmap(depth_target)
        img_merge = np.hstack([input, depth_target_col])
        
        return img_merge
    
    input = Image.open(r'D:\git\NeuLF\logs\fern_3_0_0\train\epoch-110\recon_color.png')
    depth_target = Image.open(r'D:\git\NeuLF\logs\fern_3_0_0\train\epoch-110\recon_depth.png')
    image_viz = merge_into_row(input, depth_target)
    print(image_viz.min())
    print(image_viz.max())

    plt.figure()
    plt.imshow(image_viz)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()



