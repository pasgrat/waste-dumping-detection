
# Illegal Waste Dumping Detection in Video Surveillance using Vision Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code for the university thesis "Illegal waste dumping detection in video surveillance using Vision Language Models". The project evaluates the zero-shot capabilities of state-of-the-art Vision Language Models (VLMs) for detecting and temporally localizing illegal garbage dumping events in surveillance videos.


## 🚀 Demo

https://github.com/user-attachments/assets/562ef450-5a05-4a53-ad56-050f59c702b9

## 🎯 Problem Statement

Traditional computer vision methods struggle to reliably detect illegal garbage dumping. The core challenges are:
1.  **Context Dependency:** an object is only "garbage" when it is abandoned in an inappropriate context. An object detector cannot infer this intent.
2.  **Temporal Complexity:** dumping is an action that unfolds over time (person arrival, dumping action, person departure), which is difficult to capture with frame-by-frame analysis.
3.  **Class Rigidity:** it is impossible to train a model on every possible type of object that can be dumped.

This project explores whether modern VLMs, with their ability to reason about visual data using natural language, can overcome these limitations.


## ✨ Approach

This project leverages the **Visual Question Answering (VQA)** capabilities of pre-trained VLMs in a **zero-shot** setting. Instead of training a specialized model, we treat the VLM as an AI assistant and "ask" it questions about the video content.

The process is a two-stage conversational pipeline:
1.  **Event Detection:** the model is asked a direct question, such as: *"Is a person seen littering or dumping garbage in a place where it is not allowed?"*
2.  **Temporal Localization:** if the answer is yes, a follow-up question is asked, such as: *"When does the littering or garbage dumping act start?"*

This language-guided approach allows the model to use its vast pre-trained knowledge to identify the semantic concept of "dumping" without any task-specific training.


## 🔑 Key Features

- **Zero-Shot Inference:** no model training or fine-tuning is required.
- **Contextual Understanding:** goes beyond simple object detection to interpret actions and context.
- **Temporal Localization:** pinpoints the start time of the event within the video.
- **Comparative Analysis:** evaluates four different models from two leading VLM families (`VideoLLaMA-3` and `Qwen2.5-VL`).
- **Prompt Engineering Study:** investigates how changing the textual prompt affects model performance and behavior.
- **Reproducible Pipeline:** a complete, end-to-end pipeline for video processing, inference, and evaluation is provided in a Jupyter Notebook.


## 🤖 Models Evaluated

The following state-of-the-art, open-source VLMs were used:
- [**VideoLLaMA-3**](https://github.com/DAMO-NLP-SG/VideoLLaMA3) (2B & 7B parameters)
- [**Qwen2.5-VL**](https://github.com/QwenLM/Qwen3-VL) (3B & 7B parameters)


## 🔧 Installation

1.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Run one of the two pipeline notebooks:**
    ```
    QwenVL_pipeline.ipynb
    VideoLLaMA_pipeline.ipynb
    ```
    Follow the instructions on the notebooks to install required packages.

    Key dependencies include `torch`, `transformers`, `accelerate`, `decord`, and `flash-attn`. Installing `flash-attn` may require specific CUDA versions.


## 🚀 Usage

1.  **Dataset Setup:**
    - Make sure the dataset you are using is placed in the project's root directory.
    - The expected folder structure is:
      ```
      waste-dumping-detection/
      ├── DATASET/
      │   ├── videos/
      │   │   ├── vid001.mp4
      │   │   └── ...
      │   └── labels/
      │       ├── vid001.txt
      │       └── ...
      ├── QwenVL_pipeline.ipynb
      ├── VideoLLaMA_pipeline.ipynb
      └── ...
      ```

2.  **Configure the Experiment:**
    - In the second cell of the pipeline, configure the parameters for your run, including:
      - `base_path`: the path to the dataset.
      - `model_name`: the Hugging Face path to the VLM you want to use.
      - `ideal_fps`: the target maximum frame rate for video sampling.
      - `max_chunk_size`: the maximum number of frames in a chunk.
      - `min_pixels`, `max_pixels`: the limits for the number of pixels in a frame, where applicable.
      - `question1`, `question2`: the textual prompts to be used in the event detection and temporal localization tasks, respectively.

3.  **Run the Pipeline:**
    - Execute the cells in the notebook sequentially. The notebook will handle data loading, video processing, model inference, and final evaluation. VLMs are memory hungry, make sure to have enough resources!


## 📊 Dataset

The custom dataset on which the project has been evaluated consists of **758 video clips**, balanced between positive and negative samples:
- **387 Positive Clips:** real-world instances of illegal garbage dumping and littering.
- **371 Negative Clips:** normal surveillance scenarios without any target events.

Each video in `data/videos/` has a corresponding label file in `data/labels/`. An empty `.txt` file indicates a negative sample, while a file containing a single integer represents a positive sample, with the integer being the start time of the event in seconds.

The full dataset is not provided, this repository only contains a mock dataset with a few test videos.


## 📈 Results

The experiments confirmed the high efficacy of the zero-shot VLM approach. A key finding was the distinct behavioral trade-off between the model families.

#### Performance with the Standard Prompt (5 FPS)

| Model               | Precision |   Recall   | F1-Score | MTE (seconds) |
| ------------------- | :-------: | :--------: | :------: | :-----------: |
| **VideoLLaMA-3 2B** |   0.65    |  **0.87**  |   0.74   |     17.91     |
| **VideoLLaMA-3 7B** |   0.75    |    0.81    |   0.78   |     11.60     |
| **Qwen2.5-VL 3B**   |   0.82    |    0.77    | **0.79** |     12.59     |
| **Qwen2.5-VL 7B**   | **0.91**  |    0.37    |   0.53   |    **8.43**   |

- **Qwen models** excelled in **precision**, making them suitable for automated systems requiring high confidence.
- **VideoLLaMA models** showed superior **recall**, making them ideal for flagging potential events for human review.

<img width="500" alt="Results sample charts" src="https://github.com/user-attachments/assets/79ae6cf0-d972-4381-bbac-fa7975f3f36a" />


## 📫 Contact

Pasquale Grattacaso

- **LinkedIn**: [Pasquale Grattacaso](https://linkedin.com/in/pasqualegrattacaso)
- **GitHub**: [pasgrat](https://github.com/pasgrat)
