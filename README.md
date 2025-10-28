# EVUD-benchmark
NPTEL Video Extraction Benchmark (LEARNet)

This repository provides a benchmark-ready implementation of the NPTEL Video Extraction Module, a foundational component of the LEARNet and EVUD-2M educational video understanding pipelines. The code automates large-scale data collection from the NPTEL repository, enabling reproducible benchmarking for instructional video analysis.

🔍 Overview

The pipeline performs systematic sampling, metadata cleaning, and automated video retrieval from educational course repositories. It generates a balanced and standardized dataset that supports tasks such as:

Educational video segmentation

Keyframe and semantic content extraction

Visual Table of Contents (ToC) generation

⚙️ Features

Balanced Sampling: Selects one-fourth of courses per discipline to ensure domain diversity.

Automated Web Extraction: Configures Selenium and ChromeDriver to navigate and scrape course video data.

Metadata Normalization: Cleans, merges, and exports structured CSV files for downstream LEARNet modules.

Benchmark-Ready: Designed for reproducible and scalable dataset creation across educational video domains.

🧩 Pipeline Summary

Load Metadata: Reads courses.csv containing all NPTEL course information.

Sample Courses: Randomly selects ¼ of courses per discipline for balanced coverage.

Set Up Browser Automation: Installs and configures ChromeDriver for Selenium-based scraping.

Extract Video Data: Navigates to course pages, captures URLs, titles, and video metadata.

Export Results: Outputs cleaned dataset (final_selected_rows.csv) for benchmark use.

📊 Applications

LEARNet: Learning Entropy-Aware Representation Network

EVUD-2M Benchmark Dataset

Automated Table of Content generation

Educational multimedia retrieval
