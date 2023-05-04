import torch
import numpy as np
import PIL.Image as Image


def depth2pcd(depth, disp=True, K=None):
    if K is None:
        print(
            "K is None, using default (nuscene) camera intrinsic for pcd reconstruction."
        )
        K = [[1272.5979470598488, 0.0, 826.6154927353808],
             [0.0, 1272.5979470598488, 479.75165386361925], [0.0, 0.0, 1.0]]

    if disp:
        print("Transfer disparity into depth.")
        depth = 1 / (depth + 1e-6)

    K = np.array(K)

    im_x = np.linspace(0, depth.shape[0] - 1, depth.shape[0])
    im_y = np.linspace(0, depth.shape[1] - 1, depth.shape[1])

    uv = np.meshgrid(im_x, im_y)

    pix_coord = np.stack([uv[0], uv[1], np.ones_like(uv[0])], axis=2)
    pix_coord = pix_coord.reshape(-1, 3)

    depth = depth.T.reshape(1, -1)

    cam_coord = np.multiply(depth, np.linalg.inv(K) @ pix_coord.T)

    return cam_coord.T


def export_pcd_to_ply(xyzs, rgbs=None, fileName=None):

    if fileName is None:
        print("Error, please input .ply filename for saving")
        return

    if rgbs is not None:
        assert (xyzs.shape == rgbs.shape)

    with open(fileName, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment PCL generated\n')
        f.write('element vertex {}\n'.format(len(xyzs)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyzs)):
            if rgbs is None:
                r, g, b = 1, 1, 1
            else:
                r, g, b = rgbs[i].tolist()
            x, y, z = xyzs[i].tolist()
            f.write('{} {} {} {} {} {}\n'.format(x, y, z, r, g, b))


def main():
    depth_path = '/home/edwardzhu/codes/reconstruction/explicit/test_data/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404_disp.npy'
    img_path = '/home/edwardzhu/codes/reconstruction/explicit/test_data/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg'
    # read depth
    depth = np.load(depth_path)
    im_w = depth.shape[3]
    im_h = depth.shape[2]
    depth = depth.transpose((3, 2, 1, 0)).reshape(im_w, im_h)
    # read image
    img = Image.open(img_path)
    img = img.resize((im_w, im_h))
    img = np.array(img)
    rgbs = img.reshape(-1, 3)

    pcd = depth2pcd(depth, disp=True)
    save_path = './pcd.ply'

    export_pcd_to_ply(pcd, rgbs, save_path)


if __name__ == "__main__":
    main()
