# Answer Sheet Extractor

A Streamlit application to extract register numbers and subject codes from answer sheet images using YOLO for detection and CRNN models for text extraction.

## Prerequisites
- Python 3.8+
- CUDA-enabled GPU (optional, for faster processing)
- Model files: `weights.pt`, `best_crnn_model(git).pth`, `best_subject_model_final.pth` (not included in repository due to size)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/d-kavinraja/answer-sheet-extractor.git
   cd answer-sheet-extractor
