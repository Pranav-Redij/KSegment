import streamlit as st
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =====================================================
# 1️⃣ Functions
# =====================================================

def compute_ch_value(pixels, labels, centroids):
    n = len(pixels)
    k = len(centroids)
    overall_mean = np.mean(pixels, axis=0)
    Wk = 0.0
    Bk = 0.0
    for i in range(k):
        cluster_points = [pixels[j] for j in range(n) if labels[j] == i]
        n_i = len(cluster_points)
        if n_i > 0:
            for p in cluster_points:
                Wk += np.linalg.norm(p - centroids[i]) ** 2
            Bk += n_i * (np.linalg.norm(centroids[i] - overall_mean) ** 2)
    if Wk == 0 or k <= 1 or k >= n:
        return 0.0
    return (Bk / (k - 1)) / (Wk / (n - k))


def kmeans_segmentation(img, k=5, max_iter=6, st_display=None):
    bands, height, width = img.shape
    pixels = img.reshape(bands, -1).T
    np.random.seed(42)
    random_idx = np.random.choice(pixels.shape[0], k, replace=False)
    centroids = pixels[random_idx]

    for step in range(max_iter):
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            pixels[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        centroids = new_centroids
        if st_display:
            st_display.write(f"Iteration {step+1} of {max_iter} for k={k} completed")

    segmented = labels.reshape(height, width)
    ch_value = compute_ch_value(pixels, labels, centroids)
    return segmented, ch_value

def load_image(file):
    src = rio.open(file)
    img = src.read()
    return img

# =====================================================
# 2️⃣ Streamlit GUI
# =====================================================

st.title("Multispectral Image K-Means Segmentation")

# Upload image
uploaded_file = st.file_uploader("Upload a multispectral image (TIFF/JP2)", type=["tif","jp2"])
if uploaded_file is not None:
    img = load_image(uploaded_file)
    st.success("Image loaded successfully!")

    # Show RGB preview (first 3 bands)
    if img.shape[0] >= 3:
        rgb = np.stack([img[0], img[1], img[2]], axis=-1)
        rgb_norm = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)
        st.image(rgb_norm, caption="RGB Preview", use_container_width=True)
    else:
        st.image(img[0], caption="First Band Preview", use_container_width=True)

    # Slider for max K
    max_k = st.slider("Select maximum K (number of clusters)", min_value=2, max_value=6, value=3)

    # Button to run K-Means
    if st.button("Run K-Means Segmentation"):
        st.info(f"Starting K-Means with max_k={max_k} ...")

        segmented_list = []
        ch_values = []

        # Run K-Means for k=1..max_k
        for i in range(max_k):
            k = i + 1
            seg, ch = kmeans_segmentation(img, k=k, max_iter=6, st_display=st)
            segmented_list.append(seg)
            ch_values.append(ch)
            st.write(f"CH value for k={k}: {ch}")

        # Plot CH vs K
        plt.figure(figsize=(6,4))
        plt.plot(range(1, max_k+1), ch_values, marker='o')
        plt.title("Calinski-Harabasz Index vs Number of Clusters (k)")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("CH Index")
        plt.grid(True)
        st.pyplot(plt)

        # Show only the segmented image with maximum CH value
        best_idx = np.argmax(ch_values)
        best_seg = segmented_list[best_idx]
        best_k = best_idx + 1
        st.write(f"Segmented Image with Best CH Value (k={best_k}, CH={ch_values[best_idx]:.2f})")

        # Color mapping
        custom_colors = ['yellow', 'red', 'green', 'blue', 'purple', 'orange']
        cmap = mcolors.ListedColormap(custom_colors[:best_k])

        plt.figure(figsize=(6,6))
        plt.imshow(best_seg, cmap=cmap)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("best_segmented.png")
        st.image("best_segmented.png", caption=f"Best Segmented Image (k={best_k})", use_container_width=True)



#THIS BELOW IS BRUTE FORCE JUST TO UNDERSTAND THE LOGIC

# import random
# import math

# def kmeans_segmentation(img, k=5, max_iter=10, scale_pixels=True):
#     """
#     Simple K-Means segmentation (no NumPy, no list comprehension)
#     """
#     # Get image dimensions
#     bands = len(img)
#     height = len(img[0])
#     width = len(img[0][0])

#     # (a) Convert image to pixel array
#     pixels = []
#     for i in range(height):
#         for j in range(width):
#             pixel = []
#             for b in range(bands):
#                 value = float(img[b][i][j])
#                 if scale_pixels:
#                     value = value / 10000.0
#                 pixel.append(value)
#             pixels.append(pixel)

#     num_pixels = len(pixels)

#     # (b) Initialize random centroids
#     random.seed(42)
#     centroids = []
#     for idx in random.sample(range(num_pixels), k):
#         centroids.append(pixels[idx])

#     # (c) Main K-means loop
#     for step in range(max_iter):
#         labels = []

#         # Assign each pixel to nearest centroid
#         for pixel in pixels:
#             distances = []
#             for centroid in centroids:
#                 total = 0.0
#                 for b in range(bands):
#                     total += (pixel[b] - centroid[b]) ** 2
#                 dist = math.sqrt(total)
#                 distances.append(dist)
#             # Find nearest centroid
#             min_index = 0
#             min_value = distances[0]
#             for idx in range(1, len(distances)):
#                 if distances[idx] < min_value:
#                     min_value = distances[idx]
#                     min_index = idx
#             labels.append(min_index)

#         # Update centroids
#         new_centroids = []
#         for i in range(k):
#             cluster_pixels = []
#             for j in range(num_pixels):
#                 if labels[j] == i:
#                     cluster_pixels.append(pixels[j])

#             if len(cluster_pixels) > 0:
#                 mean_pixel = []
#                 for b in range(bands):
#                     band_sum = 0.0
#                     for p in cluster_pixels:
#                         band_sum += p[b]
#                     mean_pixel.append(band_sum / len(cluster_pixels))
#                 new_centroids.append(mean_pixel)
#             else:
#                 # Keep old centroid if cluster empty
#                 new_centroids.append(centroids[i])

#         centroids = new_centroids
#         print(f"Iteration {step+1}/{max_iter} completed")

#     # (d) Reshape labels to segmented image
#     segmented = []
#     idx = 0
#     for i in range(height):
#         row = []
#         for j in range(width):
#             row.append(labels[idx])
#             idx += 1
#         segmented.append(row)

#     return segmented
