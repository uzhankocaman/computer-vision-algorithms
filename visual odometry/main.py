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
    nPoints = x1.shape[0]
    A = np.zeros((nPoints, 9))

    for i in range(nPoints):
        A[i, :] = np.kron(x1[i], x2[i])

    return A


def get_rotation_translation_matrices(Rs, Ts, img1, img2):
    depth = []
    for i in range(4):
        nPoints = img1.shape[0]
    
        M = np.zeros((3*nPoints, nPoints + 1))
    
        for j in range(nPoints):
            img1_hat = skew(img1[j])
            M[3*j:3*(j+1), j] = img1_hat @ Rs[i] @ img2[i]
            M[3*j:3*(j+1), nPoints] = img1_hat @ Ts[i]
    
        # U, D, VT = np.linalg.svd(M)
        # print(VT[-1, :])
        a, b = np.linalg.eig(M.T @ M)
        depth.append(b[:-1, a.argmin()])
        gamma = b[-1, a.argmin()]
    
        # gamma is the scale of baseline (i.e. scale of scene)
        depth[i] /= gamma  # equals setting gamma == 1
        
        depth = np.array(depth)
        idx = np.where(depth>0)[0][0]
        R, T, depth = Rs[idx], Ts[idx], depth[idx]
        return R, T, depth


def plot_3d_reconstruction(X, C0, C1, R):
    colors = get_evenly_distributed_colors(12)
    print(colors)
    print(X.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(X.shape[1]):
        c = colors[i % 12] / 255
        ax.scatter(X[0][i], X[1][i], X[2][i], color=c, alpha=1)
    ax.scatter(C0[0], C0[1], C0[2], color='red', s=100, marker='x')
    ax.scatter(C1[0], C1[1], C1[2], color='blue', s=100, marker='x')
    ax.plot([0,0], [0,0], [0,2], color='red')
    ax.set_xlim([-1, 0.5])
    ax.set_ylim([-0.6, 0.4]) 
    ax.set_zlim([0, 2])
    tmp = C1 + 2*R[:, -1]/np.linalg.norm(R[:, -1])
    ax.plot([C1[0], tmp[0]], [C1[1], tmp[1]], [C1[2], tmp[2]], color='blue')
    plt.show()
    
    
if __name__ == '__main__':
    camera_intrinsics = np.array([[2759.48, 0., 1520.69],
                                  [0., 2764.16, 1006.81],
                                  [0., 0., 1.]])
    
    #necessary points for the 8-point algorithm
    #camera 1
    x1 = [953.5, 1881.5, 953.5, 1885.5, 947.5, 681.5, 1905.5, 2157.5, 1107.5, 1741.5, 1143.5, 1701.5]
    y1 = [1405.5, 1403.5, 1889.5, 1887.5, 345.5, 813.5, 351.5, 811.5, 1205.5, 1203.5, 663.5, 665.5]
    pos_img1 = zip(y1, x1)
    
    x2 = [1629.5, 2435.5, 1629.5, 2441.5, 1175.5, 883.5, 2087.5, 2287.5, 1335.5, 1937.5, 1377.5, 1907.5]
    y2 = [1385.5, 1353.5, 1913.5, 1817.5, 195.5, 707.5, 311.5, 775.5, 1159.5, 1151.5, 565.5, 607.5]
    pos_img2 = zip(y2, x2)
        
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
    
    E = U @ np.diag((1, 1, 0)) @ VT
    
    Rz1 = np.array([[0., 1., 0.],
                [-1., 0., 0.],
                [0., 0., 1.]])

    Rz2 = np.array([[0., -1., 0.],
                    [1., 0., 0.],
                    [0., 0., 1.]])
    
    R1 = U @ Rz1.T @ VT
    R2 = U @ Rz2.T @ VT
    T_hat1 = U @ Rz1 @ np.diag((1, 1, 0)) @ U.T
    T_hat2 = U @ Rz2 @ np.diag((1, 1, 0)) @ U.T
    
    T1 = np.array([-T_hat1[1, 2], T_hat1[0, 2], -T_hat1[0, 1]])
    T2 = np.array([-T_hat2[1, 2], T_hat2[0, 2], -T_hat2[0, 1]])
    
    R, T, depth = get_rotation_translation_matrices([R1, R1, R2, R2], [T1, T1, T2, T2], img_plane1, img_plane2)
    
    points3D = depth[:,np.newaxis] * (R@img_plane2.T).T + T[np.newaxis, :]
    
    plot_3d_reconstruction(points3D.T, [0,0,0], T, R)
