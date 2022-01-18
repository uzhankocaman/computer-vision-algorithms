import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import colorsys
import functools


@functools.lru_cache(5)
def get_evenly_distributed_colors(count):
    # lru cache caches color tuples
    HSV_tuples = [(x/count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x))*255).astype(np.uint8), HSV_tuples))


def visualize_matches(image1, image2, pos_img1, pos_img2):
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.imshow(image1)
    ax2 = fig.add_subplot(122)
    ax2.imshow(image2)

    colors = get_evenly_distributed_colors(12)

    for idx, (pos1, pos2) in enumerate(zip(pos_img1, pos_img2)):
        ax1.add_patch(Circle((pos1[1], pos1[0]), radius=25, color=colors[idx % 12] / 255))
        ax2.add_patch(Circle((pos2[1], pos2[0]), radius=25, color=colors[idx % 12] / 255))

    plt.show()


def skew(x):
    # this is a tiny helper function to calculate the skew-symmetric matrizes whenever cross products are needed
    return np.array([[0., -x[2], x[1]],
                     [x[2], 0., -x[0]],
                     [-x[1], x[0], 0.]])
    

def map2pixel(x, y, intrinsics):
    # map image plance on the corresponding pixel coordinates
    homogenous_coords = np.stack((x, y, np.ones((len(x),))), axis=-1)
    px_plane = np.einsum('ij,kj -> ki', intrinsics, homogenous_coords)
    return px_plane


def map2image(x, y, intrinsics):
    # map pixel correspondences on the image plane
    homogenous_coords = np.stack((x, y, np.ones((len(x),))), axis=-1)
    inv_intrinsics = np.linalg.inv(intrinsics)
    img_plane = np.einsum('ij,kj -> ki', inv_intrinsics, homogenous_coords)
    return img_plane


def create_constraint_matrix(x1, x2):
    # TODO STUDENT
    nPoints = x1.shape[0]
    A = np.zeros((nPoints, 9))

    for i in range(nPoints):
        A[i, :] = np.kron(x1[i], x2[i])

    return A


if __name__ == '__main__':
    camera_intrinsics = np.array([[2759.48, 0.,        1520.69],
                           [0.,      2764.16,   1006.81],
                           [0.,      0.,        1.]])
    
    #necessary points for the 8-point algorithm
    #camera 1
    x1 = [953.5, 1881.5, 953.5, 1885.5, 947.5, 681.5, 1905.5, 2157.5, 1107.5, 1741.5, 1143.5, 1701.5]
    y1 = [1405.5, 1403.5, 1889.5, 1887.5, 345.5, 813.5, 351.5, 811.5, 1205.5, 1203.5, 663.5, 665.5]
    points_img1 = zip(y1, x1)
    
    #camera 2
    x2 = [1629.5, 2435.5, 1629.5, 2441.5, 1175.5, 883.5, 2087.5, 2287.5, 1335.5, 1937.5, 1377.5, 1907.5]
    y2 = [1385.5, 1353.5, 1913.5, 1817.5, 195.5, 707.5, 311.5, 775.5, 1159.5, 1151.5, 565.5, 607.5]
    points_img2 = zip(y2, x2)
    
    image1 = mpimg.imread('fountain/0005.png')
    image2 = mpimg.imread('fountain/0007.png')
    
    visualize_matches(image1, image2, zip(y1, x1), zip(y2, x2))
    
    img_plane1 = map2image(x1, y1, camera_intrinsics)
    img_plane2 = map2image(x2, y2, camera_intrinsics)
    
    A = create_constraint_matrix(img_plane1, img_plane2)
    _, D, VT = np.linalg.svd(A)
    V = VT.T
    rank_8_E = V[:, -1].reshape(3, 3)
    
    [U, D, VT] = np.linalg.svd(rank_8_E)
    V = VT.T
    
    if np.linalg.det(U) < 0 or np.linalg.det(V) < 0:
        [U, D, VT] = np.linalg.svd(-rank_8_E)
        V = VT.T