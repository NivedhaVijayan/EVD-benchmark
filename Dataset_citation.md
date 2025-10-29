---

# EVUD-2M Dataset Sources and Citations

This document provides detailed information on all datasets, repositories, and external resources used in constructing the **EVUD-2M Benchmark**.

---

## 📊 Primary Educational Video Sources

### 1. NPTEL (National Programme on Technology Enhanced Learning)

* **Description:** Large-scale educational video repository containing thousands of lecture recordings across STEM and Humanities.
* **Usage:** Primary source for authentic instructional content, providing over 19,500 lecture videos and 733K keyframes.
* **Access:** [https://nptel.ac.in/](https://nptel.ac.in/)
* **Citation:**
  NPTEL: National Programme on Technology Enhanced Learning, Government of India, IIT Consortium.

---

### 2. ClassX Dataset (Stanford University)

* **Description:** Lecture video collection from 21 courses, used to provide diverse institutional formats and presentation styles.
* **Access:** [[https://classx.stanford.edu/](https://www.youtube.com/@stanfordonline/playlists)]
* **Citation:**
  ClassX: Stanford Online Lecture and Video Indexing Platform, Stanford University.

---

### 3. SlideShare-1M

* **Description:** A large-scale slide corpus used to prevent layout bias and enhance visual diversity.
* **Access:** [[https://purl.stanford.edu/mv327tb8364](https://purl.stanford.edu/mv327tb8364)]
* **Citation:**
  S. Bhattacharya et al., *SlideShare-1M: A Large-Scale Dataset of Educational Presentation Slides*, internal curated corpus, 2024.

---

## 🧩 Auxiliary Image and Graphics Datasets

### 4. TableBank

* **Description:** Image-based dataset for table detection and recognition in documents.
* **Access:** [https://github.com/doc-analysis/TableBank](https://github.com/doc-analysis/TableBank)
* **Citation:**
  Minghao Zhang et al., “TableBank: Table Benchmark for Image-based Table Detection and Recognition,” *arXiv:1903.01949*, 2019.
  DOI: [10.48550/arXiv.1903.01949](https://doi.org/10.48550/arXiv.1903.01949)

---

### 5. TableNet

* **Description:** Deep learning dataset and model for end-to-end table detection and structure recognition in document images.
* **Access:** [https://github.com/DevashishPrasad/TableNet](https://github.com/DevashishPrasad/TableNet)
* **Citation:**
  P. Prasad et al., *TableNet: Deep Learning Model for End-to-End Table Detection and Tabular Structure Recognition*, *2019 International Conference on Document Analysis and Recognition (ICDAR)*.
  DOI: [10.1109/ICDAR.2019.00133](https://doi.org/10.1109/ICDAR.2019.00133)

---

### 6. Roboflow Educational Image Datasets

* **Description:** Platform for curated and preprocessed computer vision datasets, including educational figures, diagrams, and table images, suitable for training detection models. “From Roboflow, we leveraged curated datasets covering key domains such as table detection, figure and chart recognition, workflow and activity diagrams, and other educational diagrammatic visuals to enhance EVUD-2M’s graphical content coverage"
* **Access:** [https://roboflow.com/](https://roboflow.com/)
* **Citation:**
  Roboflow Inc., *Roboflow Datasets and Tools for Computer Vision*, 2020–2025.
  DOI / Reference: Available per individual dataset on Roboflow portal.

---

### 7. STDW – Scientific Table and Diagram Warehouse

* **Description:** Repository of scientific diagrams and charts for enhancing educational visual coverage.
* **Access:** [https://huggingface.co/datasets/n3011/STDW](https://huggingface.co/datasets/n3011/STDW)
* **Citation:**
  Z. Gao et al., *STDW: A Benchmark for Scientific Table and Diagram Understanding*, arXiv:2207.07832, 2022.
  DOI: [10.48550/arXiv.2207.07832](https://doi.org/10.48550/arXiv.2207.07832)

---

### 8. DuckDuckGo Educational Image Collection

* **Description:** Curated educational diagrams and figure samples sourced via DuckDuckGo image search for category balancing.
* **Access:** [https://duckduckgo.com/](https://duckduckgo.com/)
* ✅ Method: Searches DuckDuckGo images for a given query (e.g., “educational diagrams math”), downloads top results into a local folder, respecting safe search and timeout rules.
* ⚠️ Notes: Downloads are limited (<500 per query). Only derived features or annotations are used; images are not redistributed publicly.
* **Example code to download images from DuckDuckGo**:

```python
pip install duckduckgo-search

from duckduckgo_search import DDGS
import requests, os

query = "educational diagrams math"
output_dir = "duckduckgo_images"
os.makedirs(output_dir, exist_ok=True)

with DDGS() as ddgs:
    results = ddgs.images(keywords=query, max_results=20)
    for i, result in enumerate(results):
        url = result["image"]
        try:
            img_data = requests.get(url, timeout=5).content
            with open(f"{output_dir}/{query.replace(' ','_')}_{i}.jpg", "wb") as f:
                f.write(img_data)
            print(f"Downloaded: {url}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
```

---

## ⚙️ Access Policy

Due to dataset size limitations, **the complete EVUD-2M dataset cannot be hosted on Zenodo or GitHub**.
Only metadata CSV files and reference video links are uploaded for verification and reproducibility.
The full dataset (2M frames, 949K semantic keyframes) is **available upon reasonable academic request**.

To request dataset access, please contact the corresponding author or repository maintainer.

---

**Maintainer:** Nivedha V. V.
**Version:** v1.1 • October 2025

---


