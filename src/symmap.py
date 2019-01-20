import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from pathlib import Path
from subprocess import call
from cv2 import Canny, imread as cv_imread
from imageio import imread
from skimage.color import rgb2grey
from skimage.morphology import medial_axis, skeletonize
from skimage.transform import resize
from skimage.filters import gaussian
# from blend_modes.blend_modes import multiply as multi
from PIL import Image
from PIL.ImageChops import multiply as multi

###################### Basic image functions ##########################

def read_image(img_name, grey=False, use_opencv=False, uint8=False):
    """
    Read an image file (.png) into a numpy array in which each entry is
    a row of pixels (i.e. ``len(img)`` is the image height in px. If
    grey is True (default is False), returns a grayscale image (dtype
    uint8 if RGBA, and dtype float32 if greyscale). use_opencv uses the
    `cv2.imread` function rather than `imageio.imread`, which always
    returns a dtype of uint8. uint8 will enforce dtype of uint8 (i.e.
    for greyscale from `imageio.imread`) if set to True, but defaults
    to False.
    """
    data_dir = Path('..') / 'img'
    if use_opencv:
        if grey:
            img = cv_imread(data_dir / img_name, 0)
        else:
            img = cv_imread(data_dir / img_name)
    else:
        img = imread(data_dir / img_name, as_gray=grey)
        if uint8 and img.dtype != 'uint8':
            img = np.uint8(img)
    return img

def show_image(img, bw=False, alpha=1, no_ticks=True, title=''):
    """
    Show an image using a provided pixel row array.
    If bw is True, displays single channel images in black and white.
    """
    if not bw:
        plt.imshow(img, alpha=alpha)
    else:
        plt.imshow(img, alpha=alpha, cmap=plt.get_cmap('gray'))
    if no_ticks:
        plt.xticks([]), plt.yticks([])
    if title != '':
        plt.title = title
    plt.show()
    return

def show_original(img_name):
    """
    Debugging/development: produce and display an original image
    """
    img = read_image(img_name)
    show_image(img)
    return img

def save_image(image, figsize, save_path, ticks=False, grey=True):
    """
    Save a given image in a given location, default without ticks
    along the x and y axis, and if there's only one channel
    (i.e. if the image is greyscale) then use the gray cmap
    (rather than matplot's default Viridis).
    """
    fig = plt.figure(figsize=figsize)
    if grey:
        plt.imshow(image, cmap=plt.get_cmap('gray'))
    else:
        plt.imshow(image)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return

################# Image gradients and edge detection #############

def get_grads(img):
    """
    Convolve Sobel operator independently in x and y directions,
    to give the image gradient.
    """
    dx = ndimage.sobel(img, 0)  # horizontal derivative
    dy = ndimage.sobel(img, 1)  # vertical derivative
    return dx, dy

def get_grad(img, normalise_rgb=False):
    dx, dy = get_grads(img)
    mag = np.hypot(dx, dy)  # magnitude
    if normalise_rgb:
        mag *= 255.0 / numpy.max(mag)
    return mag

def auto_canny(image, sigma=0.4):
    """
    Zero parameter automatic Canny edge detection courtesy of
    https://www.pyimagesearch.com - use a specified sigma value
    (taken as 0.4 from Dekel et al. at Google Research, CVPR 2017)
    to compute upper and lower bounds for the Canny algorithm
    along with the median of the image, returning the edges.
    
    See the post at the following URL:
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-
    automatic-canny-edge-detection-with-python-and-opencv/
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = Canny(image, lower, upper)
    return edged

def bbox(img):
    """
    Return a bounding box (rmin, rmax, cmin, cmax). To retrieve the
    bounded region, access `image[rmin:rmax+1, cmin:cmax+1]`.
    """
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

###################### Image channel functions #########################

def to_rgb(im):
    """
    Turn a single valued array to a 3-tuple valued array of RGB, via
    http://www.socouldanyone.com/2013/03/converting-grayscale-to-rgb
    -with-numpy.html (solution 1a)
    """
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret

def to_rgba(im, alpha=255):
    """
    Turn a single valued array to a 4-tuple valued array of RGBA, with
    alpha default to 255
    """
    w, h = im.shape
    ret = np.empty((w, h, 4), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    ret[:, :, 3] = alpha
    return ret

####################### Reproducing the edge map #######################

def reproduce_edge_map():
    im = read_image('../img/tek-kimia-03-hand-hi-contrast.png', 
            grey=True, uint8=True)
    imsml = resize(im, (310, 300), anti_aliasing=True)
    imsml = np.uint8(imsml * (255/np.max(imsml)))
    for n in np.arange(120,420,20):
        p = np.invert(Canny(imsml, 100, n))
        save_image(p, (8,8), f'../img/canny/hand-edge-100-{n}.png')
    call(['convert', '-delay', '20', '../img/canny/hand-edge-*.png',
          '../img/canny/hand-edge-100-anim.gif'])
    return

############ Overlay published vs. newly calculated edge maps ##########

######################### DEPRECATED ###################################
#def overlay_edge_maps(lval=100):
#    """
#    For a given edge detection lower value (lval), retrieve the edge maps
#    stored in the img/canny directory and multiply the original published
#    edge map over them (as in photo editing software).
#
#    Default is 100, as this is the value being used to demo, but also
#    makes the function reusable if the results are not satisfactory.
#    """
#    o_path = Path(f'../img/canny/overlay{lval}/')
#    bb = [] # reuse bbox values by storing outside of the loop context
#    fg = read_image('../img/hand-edge-scaled-overlay.png')
#    for n in np.arange(200,520,20):
#        im = read_image(f'../img/canny/hand-edge-{lval}-{n}.png')
#        if bb == []:
#            bb = bbox(np.invert(im))
#        a, b, c, d = bb
#        imcrop = im[a+3:b-2, c+3:d-2]
#        m = multi(imcrop.astype(float), fg.astype(float), 0.7)
#        save_image(m, (8,8), o_path / f'hand-overlay-{lval}-{n}.png')
#    call(['convert', '-delay', '20', o_path / f'hand-overlay-{lval}-*.png',
#          o_path / f'hand-overlay-{lval}-anim.gif'])
#    return

def overlay_edge_maps(lval=100):
    """
    For a given edge detection lower value (lval), retrieve the edge maps
    stored in the img/canny directory and multiply the original published
    edge map over them (as in photo editing software).

    Default is 100, as this is the value being used to demo, but also
    makes the function reusable if the results are not satisfactory.
    """
    o_path = Path(f'../img/canny/overlay{lval}/')
    bb = [] # reuse bbox values by storing outside of the loop context
    fg = Image.open('../img/hand-edge-scaled-overlay.png')
    for n in np.arange(120,420,20):
        im = Image.open(f'../img/canny/hand-edge-{lval}-{n}.png')
        if bb == []:
            bb = bbox(np.invert(im))
        a, b, c, d = bb
        imcrop = im.crop((c+3, a+3, d-2, b-2))
        m = multi(fg, imcrop)
        m.save(o_path / f'hand-overlay-{lval}-{n}.png')
    call(['convert', '-delay', '20', o_path / f'hand-overlay-{lval}-*.png',
          o_path / f'hand-overlay-{lval}-anim.gif'])
    return

################### Medial axis/skeleton functions #####################

def auto_hand_img(l=100, u=260):
    """
    Input image for scan_hand function.
    """
    im = read_image(f'../img/canny/hand-edge-{l}-{u}.png',
                    grey=True, uint8=True)
    a, b, c, d = bbox(np.invert(im))
    imcrop = im[a+3:b-2, c+3:d-2]
    return imcrop

def medial_scan_hand(im=None, skeletonize=False, save_path=None):
    """
    Read in the hand image and run MAT on it then display next to original.
    Code via skimage plot_medial_transform example.

    If save_path is given, will write the result to a file.
    """
    if im is None:
        im = auto_hand_img()
    # Compute the medial axis (skeleton) and the distance transform
    skel, distance = medial_axis(im, return_distance=True)

    # Distance to the background for pixels of the skeleton
    dist_on_skel = distance * skel

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(im, cmap=plt.get_cmap('gray'), interpolation='nearest')
    ax1.axis('off')
    ax2.imshow(dist_on_skel, cmap=plt.get_cmap('Spectral'), interpolation='nearest')
    ax2.contour(im, [0.5], colors='w')
    ax2.axis('off')

    fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path)

def skeleton_hand(save_path=None):
    """
    Skeletonize the edge map of the hand photo, optionally saving to disk.
    """
    edged = auto_hand_img() / 255
    im2 = to_rgba(np.copy(edged) * 255)
    skel = skeletonize(edged)
    im2[skel] = [255, 0, 0, 255]
    if save_path is None:
        return im2
    else:
        save_image(im2, (8,8), save_path)
        return im2

def light_blur_skeleton_hand(save_path=None):
    """
    Skeletonize the edge map of the hand photo after applying a Gaussian
    filter (blur) to it first so as to remove the excess edges seen by the
    skeletonize algorithm. Light blur, sigma=2.
    """
    im = auto_hand_img() # reload the edge map
    blurred = gaussian(np.copy(im)) 
    #blurred = blurred * blurred # strengthen the image by multiplying
    im2 = to_rgba(np.copy(im)) # take an RGBA copy to add the skeleton onto
    skel = skeletonize(blurred) # given as a Boolean array
    skel_blur = gaussian(np.copy(skel), sigma=2)
    skel_blur *= (255/np.max(skel_blur))
    # manually examine the distribution to set a threshold for binarisation
    # for i in np.arange(0,101,1): print(np.percentile(skel_blur, i))
    skel_blur[skel_blur >= 30] = 255
    skel_blur[skel_blur < 30] = 0
    skel2 = (skel_blur/255).astype(bool)
    # also expand the edge map using the blurred version for visibility
    im2[blurred <= 0.75] = [0,0,0,255]
    # set the skeleton pixels to red in the edge map copy
    im2[skel2] = [255, 0, 0, 255]
    if save_path is None:
        return im2
    else:
        save_image(im2, (8,8), save_path)
        return im2


def heavy_blur_skeleton_hand(save_path=None):
    """
    Skeletonize the edge map of the hand photo after applying a Gaussian
    filter (blur) to it first so as to remove the excess edges seen by the
    skeletonize algorithm. Heavy blur, sigma=3.
    """
    im = auto_hand_img() # reload the edge map
    blurred = gaussian(np.copy(im)) 
    #blurred = blurred * blurred # strengthen the image by multiplying
    im2 = to_rgba(np.copy(im)) # take an RGBA copy to add the skeleton onto
    skel = skeletonize(blurred) # given as a Boolean array
    skel_blur = gaussian(np.copy(skel), sigma=3)
    skel_blur *= (255/np.max(skel_blur))
    # manually examine the distribution to set a threshold for binarisation
    # for i in np.arange(0,101,1): print(np.percentile(skel_blur, i))
    skel_blur[skel_blur >= 30] = 255
    skel_blur[skel_blur < 30] = 0
    skel2 = (skel_blur/255).astype(bool)
    # also expand the edge map using the blurred version for visibility
    im2[blurred <= 0.75] = [0,0,0,255]
    # set the skeleton pixels to red in the edge map copy
    im2[skel2] = [255, 0, 0, 255]
    if save_path is None:
        return im2
    else:
        save_image(im2, (8,8), save_path)
        return im2

def reproduce_full_figure(save_path=None, blur_skel=False):
    """
    Read in the hand image and run MAT on it then display next to original.

    If save_path is given, will write the result to a file.
    """
    im = read_image('../img/tek-kimia-03-hand-hi-contrast.png',
                    grey=True, uint8=True)
    edged = auto_hand_img() / 255
    im2 = to_rgba(np.copy(edged) * 255)
    if blur_skel:
        im2 = heavy_blur_skeleton_hand()
    else:
        skel = skeletonize(edged)
        im2[skel] = [255, 0, 0, 255]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    ax1.imshow(im, cmap=plt.get_cmap('gray'), interpolation='nearest')
    ax1.axis('off')
    ax2.imshow(edged, cmap=plt.get_cmap('gray'))
    ax2.axis('off')
    ax3.imshow(im2)
    ax3.axis('off')

    fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path)
    plt.close(fig)
    return
