🖼️ Image Segmentation using K-Means and Calinski–Harabasz Index
📘 Overview

This project performs image segmentation on multispectral or satellite images using the K-Means clustering algorithm.
It automatically finds the best number of clusters (K) based on the Calinski–Harabasz (CH) Index, which measures both within-cluster compactness and between-cluster separation — higher CH means better segmentation.

⚙️ Features

Upload multispectral or RGB images (.tif, .jp2, etc.)

Run K-Means segmentation for multiple K values

Compute Calinski–Harabasz Index to find the optimal K

Display CH vs K plot for easy visualization

View the best segmented image

Change segment colors in real time using color pickers (Streamlit UI)

🧠 How It Works

The uploaded image is converted into pixel arrays.

K-Means clustering groups pixels based on spectral similarity.

For each K (from 1 to selected max), segmentation is done and CH Index is calculated.

The K with the highest CH value is chosen as the best segmentation result.

The final segmented image is displayed with customizable colors.

🚀 How to Run
Step 1: Install dependencies
pip install streamlit rasterio matplotlib numpy

Step 2: Run the app
streamlit run k_mean.py --server.address=0.0.0.0

Step 3: Open in browser

On your PC: http://localhost:8501

On your mobile (same Wi-Fi):
http://<your-pc-ip>:8501

📊 Output Example

Graph: CH Index vs K

Text: "Best K value based on CH Index = 3"

Image: Segmented image with user-selected colors

🧩 Tech Stack

Python

Streamlit (for UI)

Rasterio (for image handling)

Matplotlib (for plots)

NumPy (for numerical computation)

🏁 Conclusion

This project helps identify the optimal K value for image segmentation using a statistical metric (CH Index), ensuring better separation and clustering of different regions in multispectral images.