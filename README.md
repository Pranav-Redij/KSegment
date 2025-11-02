# 🖼️ Image Segmentation using K-Means and Calinski–Harabasz Index

## 📘 Overview
This project performs **image segmentation** on multispectral or satellite images using the **K-Means clustering algorithm**.  
It automatically finds the **best number of clusters (K)** based on the **Calinski–Harabasz (CH) Index**, which measures both **within-cluster compactness** and **between-cluster separation** — higher CH means better segmentation.

---

## ⚙️ Features
- 📤 Upload multispectral or RGB images (`.tif`, `.jp2`, etc.)  
- 🌀 Run **K-Means segmentation** for multiple K values  
- 📈 Compute **Calinski–Harabasz Index** to find the optimal K  
- 📊 Display **CH vs K** plot for easy visualization  
- 🖼️ View the **best segmented image**  
- 🎨 Change **segment colors in real time** using color pickers (Streamlit UI)

---

## 🧠 How It Works
1. The uploaded image is converted into pixel arrays.  
2. K-Means clustering groups pixels based on spectral similarity.  
3. For each K (from 1 to selected max), segmentation is done and the CH Index is calculated.  
4. The K with the **highest CH value** is chosen as the best segmentation result.  
5. The final segmented image is displayed with **customizable colors**.

---

## 🚀 How to Run

### Step 1: Install dependencies

pip install streamlit rasterio matplotlib numpy
Step 2: Run the app
bash
Copy code
streamlit run k_mean.py --server.address=0.0.0.0
Step 3: Open in browser
💻 On your PC: http://localhost:8501

📱 On your mobile (same Wi-Fi):
http://<your-pc-ip>:8501

📊 Output Example
Graph: CH Index vs K

Text: “Best K value based on CH Index = 3”

Image: Segmented image with user-selected colors

 
# 🧩 Tech Stack

| Component   | Purpose |
|--------------|----------|
| **Python**   | Core programming language |
| **Streamlit** | Interactive web UI |
| **Rasterio** | Multispectral image handling |
| **Matplotlib** | CH Index plotting |
| **NumPy** | Numerical computations for K-Means |


# 🏁 Conclusion
This project helps identify the optimal K value for image segmentation using a statistical metric (CH Index).
It ensures better separation and clustering of different regions in multispectral images while giving users full control over visualization and color customization through a clean Streamlit interface.
