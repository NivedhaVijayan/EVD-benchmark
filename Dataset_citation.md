# EVUD-2M Dataset Sources and Citations

This document provides detailed information on all datasets, repositories, and external resources used in constructing the **EVUD-2M Benchmark**.

---

## 📊 Primary Educational Video Sources

## 1. NPTEL (National Programme on Technology Enhanced Learning)
- **Description:** Large-scale educational video repository containing thousands of lecture recordings across STEM and Humanities.
- **Usage:** Primary source for authentic instructional content, providing over 19,500 lecture videos and 733K keyframes.
- **Access:** [https://nptel.ac.in/](https://nptel.ac.in/)
- **Citation:**  
  NPTEL: National Programme on Technology Enhanced Learning, Government of India, IIT Consortium.

---

## 2. ClassX Dataset (Stanford University)
- **Description:** Lecture video collection from 21 courses, used to provide diverse institutional formats and presentation styles.  
- **Access:** [[https://classx.stanford.edu/](https://www.youtube.com/@stanfordonline/playlists)]
- **Citation:**  
  ClassX: Stanford Online Lecture and Video Indexing Platform, Stanford University.

---

## 3. SlideShare-1M
- **Description:** A large-scale slide corpus used to prevent layout bias and enhance visual diversity.  
- **Access:** [https://purl.stanford.edu/mv327tb8364]
- **Citation:**  
  S. Bhattacharya et al., *SlideShare-1M: A Large-Scale Dataset of Educational Presentation Slides*, internal curated corpus, 2024.

---

## 🧩 Auxiliary Image and Graphics Datasets

## 4. TableBank
- **Description:** Image-based dataset for table detection and recognition in documents.
- **Access:** [https://github.com/doc-analysis/TableBank](https://github.com/doc-analysis/TableBank)
- **Citation:**  
  Minghao Zhang et al., “TableBank: Table Benchmark for Image-based Table Detection and Recognition,” *arXiv:1903.01949*, 2019.  
  DOI: [10.48550/arXiv.1903.01949](https://doi.org/10.48550/arXiv.1903.01949)

---

## 5. STDW – Scientific Table and Diagram Warehouse
- **Description:** Repository of scientific diagrams and charts for enhancing educational visual coverage.
- **Citation:**
- **Access** [https://huggingface.co/datasets/n3011/STDW]
  Z. Gao et al., *STDW: A Benchmark for Scientific Table and Diagram Understanding*, arXiv:2207.07832, 2022.  
  DOI: [10.48550/arXiv.2207.07832](https://doi.org/10.48550/arXiv.2207.07832)

---

## 6. DuckDuckGo Educational Image Collection
- **Description:** Curated educational diagrams and figure samples sourced via DuckDuckGo image search for category balancing.  
- **Access:** [https://duckduckgo.com/](https://duckduckgo.com/)
- ✅ This:
Searches DuckDuckGo images for a given query (e.g., “educational diagrams math”).
Downloaded the top max_results images into a local folder.
Respects safe search and timeout rules.
⚠️ Always limit downloads (e.g., <500 per query) and we did not redistribute images publicly. We only use derived features or annotations in our dataset (EVUD-2M).
- **example code to download images from duckduckgo**
pip install duckduckgo-search

from duckduckgo_search import DDGS
import requests, os

query = "educational diagrams math"
output_dir = "duckduckgo_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize DuckDuckGo search
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
---

## ⚙️ Access Policy

Due to dataset size limitations, **the complete EVUD-2M dataset cannot be hosted on Zenodo or GitHub**.  
Only metadata CSV files and reference video links are uploaded for verification and reproducibility.  
The full dataset (2M frames, 949K semantic keyframes) is **available upon reasonable academic request**.

To request dataset access, please contact the corresponding author or repository maintainer.

---

---

**Maintainer:** Nivedha V. V.  
**Version:** v1.0 • October 2025
