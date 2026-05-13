import numpy as np
import matplotlib.pyplot as plt
import os

# MiniCV modules
import image_io as io
import utils as fdf
import filters as flt
import utils as gt 
import features as feat                


# --------------------------------------------
# 1. LOAD IMAGE
# --------------------------------------------

img = io.read_image("C:\\Users\\Mohammed_AlJamal\\OneDrive\\Desktop\\milestone 1\\minicv\\me_and_friends.jpeg", as_gray=False)

print("\nOriginal image")
print("shape:", img.shape)
print("dtype:", img.dtype)
print("min,max:", img.min(), img.max())


# --------------------------------------------
# 2. NORMALIZATION TEST
# --------------------------------------------

img_norm = fdf.normalize_image(img, "0to1")

print("\nNormalized image range:")
print(img_norm.min(), img_norm.max())


# --------------------------------------------
# 3. CONVERT TO GRAYSCALE (if RGB)
# --------------------------------------------

if img_norm.ndim == 3:

    img_gray = (
        0.299*img_norm[:,:,0] +
        0.587*img_norm[:,:,1] +
        0.114*img_norm[:,:,2]
    )

else:

    img_gray = img_norm


# --------------------------------------------
# 4. FILTERS TEST
# --------------------------------------------

mean_img = flt.mean_filter(img_gray, 3)

gaussian_img = flt.gaussian_filter(img_gray, 5, 1.2)

median_img = flt.median_filter(img_gray, 3)

laplace_img = flt.laplacian_filter(img_gray)

sharpen_img = flt.sharpening_filter(img_gray)


# --------------------------------------------
# 5. SOBEL EDGE TEST
# --------------------------------------------

gx, gy, magnitude = flt.sobel_gradients(img_gray)

print("\nGradient magnitude stats")
print("min:", magnitude.min())
print("max:", magnitude.max())


# --------------------------------------------
# 6. HISTOGRAM TEST
# --------------------------------------------

hist = flt.histogram(img_gray)

equalized_img = flt.histogram_equalization(img_gray)

print("\nHistogram sum:", hist.sum())


# --------------------------------------------
# 7. THRESHOLDING TEST
# --------------------------------------------

binary_global = flt.global_threshold(img_gray, 0.5)

binary_otsu = flt.otsu_threshold(img_gray)

binary_adaptive = flt.adaptive_threshold(img_gray, 7, 3)


# --------------------------------------------
# 8. BIT PLANE TEST
# --------------------------------------------

bit_plane_7 = flt.bit_plane_slice(img_gray, 7)


# --------------------------------------------
# 9. FEATURE STYLE METRICS (basic)
# --------------------------------------------

mean_intensity = np.mean(img_gray)

edge_density = np.sum(magnitude > 0.2) / magnitude.size

print("\nFeature style metrics:")
print("mean intensity:", mean_intensity)
print("edge density:", edge_density)


# --------------------------------------------
# 10. DISPLAY RESULTS
# --------------------------------------------

plt.figure(figsize=(12,10))


plt.subplot(4,4,1)
plt.title("Original")
plt.imshow(img_gray, cmap="gray")
plt.axis("off")


plt.subplot(4,4,2)
plt.title("Mean")
plt.imshow(mean_img, cmap="gray")
plt.axis("off")


plt.subplot(4,4,3)
plt.title("Gaussian")
plt.imshow(gaussian_img, cmap="gray")
plt.axis("off")


plt.subplot(4,4,4)
plt.title("Median")
plt.imshow(median_img, cmap="gray")
plt.axis("off")


plt.subplot(4,4,5)
plt.title("Sobel magnitude")
plt.imshow(magnitude, cmap="gray")
plt.axis("off")


plt.subplot(4,4,6)
plt.title("Laplacian")
plt.imshow(laplace_img, cmap="gray")
plt.axis("off")


plt.subplot(4,4,7)
plt.title("Sharpen")
plt.imshow(sharpen_img, cmap="gray")
plt.axis("off")


plt.subplot(4,4,8)
plt.title("Histogram equalized")
plt.imshow(equalized_img, cmap="gray")
plt.axis("off")


plt.subplot(4,4,9)
plt.title("Global threshold")
plt.imshow(binary_global, cmap="gray")
plt.axis("off")


plt.subplot(4,4,10)
plt.title("Otsu threshold")
plt.imshow(binary_otsu, cmap="gray")
plt.axis("off")


plt.subplot(4,4,11)
plt.title("Adaptive threshold")
plt.imshow(binary_adaptive, cmap="gray")
plt.axis("off")


plt.subplot(4,4,12)
plt.title("Bit plane 7")
plt.imshow(bit_plane_7, cmap="gray")
plt.axis("off")


plt.subplot(4,4,13)
plt.title("Histogram")
plt.plot(hist)


plt.tight_layout()

plt.show()


# --------------------------------------------
# 11. CONVOLUTION SPEED TEST
# --------------------------------------------

print("\nTesting convolution speed...")

test_kernel = np.ones((5,5))/25

filtered = fdf.apply_filter(img_gray, test_kernel)

print("Filtering completed successfully")


# --------------------------------------------
# END
# --------------------------------------------

print("\nMiniCV test completed successfully")



# --------------------------------------------
# 12. SAVE OUTPUT IMAGES
# --------------------------------------------


OUTPUT_DIR = "output_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)


plt.imsave(f"{OUTPUT_DIR}/01_original.jpeg", img_gray, cmap="gray")

plt.imsave(f"{OUTPUT_DIR}/02_mean.jpeg", mean_img, cmap="gray")

plt.imsave(f"{OUTPUT_DIR}/03_gaussian.jpeg", gaussian_img, cmap="gray")

plt.imsave(f"{OUTPUT_DIR}/04_median.jpeg", median_img, cmap="gray")

plt.imsave(f"{OUTPUT_DIR}/05_sobel.jpeg", magnitude, cmap="gray")

plt.imsave(f"{OUTPUT_DIR}/06_laplacian.jpeg", laplace_img, cmap="gray")

plt.imsave(f"{OUTPUT_DIR}/07_sharpen.jpeg", sharpen_img, cmap="gray")

plt.imsave(f"{OUTPUT_DIR}/08_hist_equalized.jpeg", equalized_img, cmap="gray")

plt.imsave(f"{OUTPUT_DIR}/09_global_threshold.jpeg", binary_global, cmap="gray")

plt.imsave(f"{OUTPUT_DIR}/10_otsu_threshold.jpeg", binary_otsu, cmap="gray")

plt.imsave(f"{OUTPUT_DIR}/11_adaptive_threshold.jpeg", binary_adaptive, cmap="gray")

plt.imsave(f"{OUTPUT_DIR}/12_bit_plane_7.jpeg", bit_plane_7, cmap="gray")


# save histogram figure

plt.figure()
plt.plot(hist)
plt.title("Histogram")

plt.savefig(f"{OUTPUT_DIR}/13_histogram.jpeg")

plt.close()

print("\nAll images saved inside folder:", OUTPUT_DIR) 