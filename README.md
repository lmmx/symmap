# symmap
Reproducing symmetry maps from Tek &amp; Kimia (2003) [Symmetry Maps of Free-Form Curve Segments via Wave Propagation](https://doi.org/10.1023/A:1023753317008) (presented at ICCV 1999)

Specifically I want to reproduce figure 5 (parts c-e), below

![](img/tek-kimia-03_hand.png)

The algorithm for computing the medial axis transform is outlined in detail
[here](https://stackoverflow.com/a/52796778/2668831), and is implemented
in `scikit-image` as
[`skimage.morphology.medial_axis`](http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.medial_axis)
(as well as 
[`skimage.morphology.skeletonize`](http://scikit-image.org/docs/dev/api/skimage.morphology.html#skeletonize)).

## Reproducing the edge map

`auto_canny` did not work well, and I'm pretty sure that using a smaller image gave better results.

Running `reproduce_edge_map()` does the following:

- Resize the image to a smaller size (300wx310h).
  - This produces decimal values (there's an interpolation step) so normalise to
    0-255 and convert back to uint8 (needed for OpenCV's Canny edge detector).
- Save a range of upper bounds for Canny edge detection (trial and error testing showed a lower bound
  of 60-100 was good) and produce an animation cycling through them to visualise what change they produce.

```py
im = read_image('../img/tek-kimia-03-hand-hi-contrast.png', grey=True, uint8=True)
imsml = resize(im, (310, 300), anti_aliasing=True)
imsml = np.uint8(imsml * (255/np.max(imsml)))
for n in np.arange(200,520,20):
    save_image(np.invert(Canny(imsml, 100, n)), (8,8), f'../img/canny/hand-edge-100-{n}.png')
call(['convert', '-delay', '20', '../img/canny/hand-edge-*.png', '../img/canny/hand-edge-100-anim.gif'])
```

![](img/canny/hand-edge-100-anim.gif)

The question next becomes how to determine the edge map closest to that in the original paper.

- Firstly, we must crop the box matplotlib put around the edge map
  - From inspecting the box, it seems the border is a black pixel line with a grey pixel either side, so
    incrementing the rmin/cmin and decrementing rmax/cmax by 3 from the usual formula of
    `image[rmin:rmax+1, cmin:cmax+1]` gives a formula of `image[rmin+3:rmax-2, cmin+3:cmax-2]`.

```py
hand200 = read_image('../img/canny/hand-edge-100-200.png', grey=True, uint8=True)
hand_edged = read_image('../img/tek-kimia-03_hand-edged.png', grey=True, uint8=True)
a, b, c, d = bbox(np.invert(hand200)) # note that these values are reusable
hand200crop = hand200[a+3:b-2, c+3:d-2]
```

Now the 2 images must be scaled to the same size, so as to compare their edges. Doing so in desktop editing
software shows it's possible, but not immediately clear how to proceed.

![](img/hand-edge-100-200_manual-edge-overlay.png)

Because life is short and this isn't exactly the point of this exercise, I'm just going to export this scaled
layer and use this for comparison. To reproduce the above image with the rest of those shown in the animation
above, I loaded the python package `blend_modes` which implements a `multiply` function equivalent to that used
in photo editing software _but it didn't work..._

```py
from blend_modes import multiply as multi
# this package requires RGBA
hand200cropRGBA = to_rgba(hand200crop)
handscaled = read_image('../img/hand-edge-scaled-overlay.png')
multiplied = multi(hand200cropRGBA.astype(float), handscaled.astype(float), 1.0)
```

`PIL.ImageChops` also implements this, but switching to PIL's `Image` class require a bit of rewriting.

In PIL, cropping takes `(cmin, rmin, cmax, rmax)` rather than numpy's `[rmin, rmax, cmin, cmax]` which is
given as `[a, b, c, d]` from the `bbox` function, such that cropping with PIL becomes:

```py
cropped = im_pil.crop((c, a, d, b))
```

Though here the increments/decrements remain the same as above: `a+3, b-2, c+3, d-2`.

```py
from PIL import Image
from PIL.ImageChops import multiply as multi
fg = Image.open('../img/hand-edge-scaled-overlay.png')
im = Image.open('../img/canny/hand-edge-100-200.png').crop((c+3, a+3, d-2, b-2))
blended = multiply(fg, im)
# blended.show() or blended.save() will display the result
```

![](img/demo_mult_blended.png)

Doing this for all the edge maps calculated for the lower Canny threshold of 100, you can make an animation
showing the discrepancies from the published image, as below:

![](img/canny/overlay100/hand-overlay-100-anim.gif)

Looking through these, I chose the upper threshold of 260 as retaining most of the most important edges from
the original publication (noting that some of those that are lost at this level indicate vein shadows rather
than solid features).

## Reproducing the symmetry map

The last thing to do now is get the symmetry map for the edge map with Canny thresholds 100-260.

The `scan_hand` function takes the aforementioned edge map and runs the code given in the scikit-image
[`plot_medial_transform` example](http://scikit-image.org/docs/0.10.x/auto_examples/plot_medial_transform.html).

```py
scan_hand()
```

![](img/hand-medial-transform.png)

It's clearly visible here that the ridges demarcate the symmetry map, which can be made explicit by running
`skimage.morphology.skeletonize` instead (the documentation notes that the two functions are actually distinct)

The last thing to do to finish this off is to calculate the skeleton and display it overlayed on the symmetry
map as in the original publication.

The `skeleton_hand` function does the following to produce this overlay:

```py
im = auto_hand_img() # reload the edge map
im2 = to_rgba(np.copy(im) * 255) # take an RGBA copy to add the skeleton onto
skel = skeletonize(im) # given as a Boolean array
im2[skel] = [255, 0, 0, 255] # set the skeleton pixels to red in the edge map copy
```

![](img/hand-skeleton.png)

I think the artifactual 'spider web'-like extra lines are probably due to using a too-low
resolution image, such that 'edges' are overdetected. To finish up, re-run with a larger image
and experimenting with Gaussian blur's sigma value to reproduce the existing edge map:

```py
im = read_image('../img/tek-kimia-03-hand-hi-contrast.png', grey=True, uint8=True)
p = np.invert(Canny(im, 100, 260))
save_image(p, (8,8), f'../img/large-edge-100-260.png')
from skimage.filters import gaussian
blurred = gaussian(im, sigma=2)
blurred *= (255/np.max(blurred))
edged = np.invert(Canny(blurred.astype(np.uint8), 50, 100))
im2 = to_rgba(np.copy(edged))
skel = skeletonize(edged / 255)
im2[skel] = [255, 0, 0, 255]
```

Unfortunately this just makes the spidery banding worse!

![](img/spidery-banded-skeleton.png)

Lastly, the function `reproduce_full_figure` will display the 3 panels of the original figure:

![](img/reproduced-figure.png)

## Postscript - resolving the spidery banding

As a first attempt to remove the excess symmetry lines, I'll try smoothing the edge map first with
a Gaussian filter (sigma=3) and manually examining and setting thresholds at 30/255 on the skeleton
and 75% on the blurred edge map (i.e. thresholds for their binarisation).

```py
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
# expand the edge map a little for visibility
im2[blurred2 <= 180] = [0,0,0,255]
im2[skel2] = [255, 0, 0, 255] # set the skeleton pixels to red in the edge map copy
```

![](img/blurred-skeletonized-hand.png)

That looks much better (but note the symmetry lines don't fully touch the edge map any more). As a
final clean up, I'm going to change the sigma value to 2 on the Gaussian smoothing filter.
