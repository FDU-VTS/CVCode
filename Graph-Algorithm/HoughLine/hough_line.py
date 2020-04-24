import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.morphology
import skimage.transform
import skimage.feature
import copy

original_img = skimage.io.imread("cars.png")
img = copy.deepcopy(original_img)
h, w, _ = img.shape
for i in range(h):
    for j in range(w):
        r, g, b, a = img[i, j]
        if r < 180 or g < 180 or b < 180:
            img[i, j] = [0, 0, 0, 255]
img = skimage.color.rgb2gray(skimage.color.rgba2rgb(img))
img = skimage.morphology.dilation(img)
img = skimage.feature.canny(img)
lines = skimage.transform.probabilistic_hough_line(img)
for line in lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]), color='red')
plt.imshow(original_img)
plt.show()
