**LEARNet Framework and EVUD-2M Benchmark**
**Overview**

The LEARNet architecture provides an end-to-end system for large-scale educational video understanding, combining the Temporal Information Bottleneck (TIB) and the Hierarchical Educational Scene Parser (SSD) into a unified workflow. This system has been leveraged to construct the EVUD-2M benchmark, a large-scale, fine-grained dataset of educational videos.
This work includes novel computational code developed for the EVUD-2M benchmark and educational video understanding. 

Modules:

tib_module.py — Temporal Information Bottleneck (TIB) for keyframe extraction

ssd_module.py — Spatial-Semantic Decoder (SSD) with relational verification network (RCVN)

**Workflow**

Temporal Information Bottleneck (TIB):

    Analyzes raw video sequences to identify pedagogically significant keyframes.
    Reduces data volume by over 70%, selecting ~949,000 keyframes from 2 million frames.
    Hierarchical Educational Scene Parser (SSD with RCVN):
    Transforms curated keyframes into semantically enriched educational content.
    Uses a Relational Verification Network (RCVN) to propagate fine-grained region-level annotations efficiently.
    Ensures high annotation consistency across the dataset.
    EVUD-2M Benchmark Highlights
    Temporal coherence: Keyframes capture the most pedagogically informative moments.
    Spatial semantic richness: Region-level annotations for slides, diagrams, equations, and handwritten content.
    Structural diversity: Covers various instructional formats ensuring broad coverage of educational scenarios.

**Implementation Details**

Language & Frameworks: Python 3.9, PyTorch 2.1.0

Integrated Pipeline: TIB → SSD → Scene Graph → Annotated Dataset

Relational Verification Network: Batch normalization ensures stable learning across diverse educational content types.

Training & Validation: 5-fold cross-validation with 80/20 train-test split, Adam optimizer (learning rate = 0.001, batch size = 32).

Hardware: Dual NVIDIA RTX 4090 GPUs, 48GB VRAM, 128GB RAM, 24-core CPU.

**Reproducibility and Scalability**

This framework allows efficient processing of large-scale educational video corpora while maintaining high-quality annotations. The EVUD-2M benchmark establishes a reproducible foundation for future research in educational video understanding.
EVUD-2M Benchmark Details

**The EVUD-2M benchmark includes:**

The EVUD-2M benchmark was created by extracting keyframes from a large collection of educational videos (~2 million frames).
Size Note: The dataset is too large to host on GitHub or Zenodo.
Access: Interested researchers can request access by contacting the authors.
Metadata CSV available in zenodo "https://zenodo.org/records/14036059?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjFhYzEwNWJjLWM3ZjAtNDQ3YS05MjVlLWJmNjBiM2JiYzJjOSIsImRhdGEiOnt9LCJyYW5kb20iOiI3YjY3MTc2NjMxZTY5YmY5Mzk1NzgwZmRmNDY4YzU3NyJ9.6UNbgftncY4jDkoK_OCt-WS_9Ph5iiB5N8K1lAsQHKwD0RSOmUB2qZ7huptLUw90S69YteDvXuC-ysjoS0XQ9g": The repository includes a CSV listing the original video links, along with course and lecture details (Discipline, Subject ID, Subject Name, Institute, Type, Coordinator Name, Unique Display Names). 

**Note:** Full-scale reproduction of EVUD-2M requires downloading videos from the links in the CSV and running the TIB & SSD pipeline provided in this repository.
Note: The full EVUD-2M dataset (~2 million frames) is too large to host on GitHub or Zenodo. A CSV with links to original NPTEL/ClassX videos is included. Full dataset access is available on request.
