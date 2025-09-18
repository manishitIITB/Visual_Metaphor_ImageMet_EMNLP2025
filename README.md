# ImageMet: a Visual Metaphor Dataset
This is the official repository for our EMNLP 2025 paper: “Looking Beyond the Pixels: Evaluating Visual Metaphor Understanding in VLMs.”

![Examples from the ImageMet Dataset](https://github.com/user-attachments/assets/9829707b-727b-4e55-b56b-07c34fe0eef3)


## 📂 Repository Structure 

```
.
├── data/  
│   ├── elaborations.pkl        # Pickled list of elaborations for image generation
│   └── image_generation.py     # Script for generating images from elaborations

├── prompts/  
│   ├── scene_graph_generation.txt       # Prompt template for scene graph generation  
│   ├── visual_elaboration_from_similes.txt  # Prompt template for elaboration from similes  
│   └── visual_metaphor_analysis.txt     # Prompt template for metaphor analysis  

├── src/  
│   ├── llava/  
│   │   ├── collators.py        # Collation utilities for data batching  
│   │   ├── dataset.py          # Dataset loading and preprocessing  
│   │   └── utils.py            # Training Utils 
│   │  
│   └── qwen/  
│       └── qwen_finetune.py    # Fine-tuning script for Qwen models (unsloth)

``` 


## Dataset: ImageMet

ImageMet has two main objectives:

1. Supply synthetic training and validation sets for fine-tuning models, enabling them to learn structured metaphorical comparisons.
2. Provide a challenging, manually annotated test set with clearly intended visual metaphors, minimizing ambiguity and subjectivity.

### Dataset Design

Captions follow a simile-based template:

```
<Primary Concept> is as <Attribute> as <Secondary Concept>
```

This explicit decomposition encourages models to verbalize metaphorical reasoning by identifying the Primary (target domain), Secondary (source domain), and their shared Attribute.


### Training/Validation Sets:

You can find the dataset [here](https://huggingface.co/datasets/manishitkundu/imagemet).

Information about the data:

1. Synthetically generated (Image, Simile) pairs.
2. Primary concepts curated (high concreteness/imageability nouns)
3. Attributes Extracted from WordNet and LLM prompts
4. Secondary Concepts generated with GPT-4o and filtering by semantic dissimilarity and perplexity.
5. Synthetic images with Stable Diffusion 3.5 Large from GPT-4o-generated visual elaborations.

For more details, please go through our paper.

### Test Set:

Note: Currently, we cannot release the test set due to copyright constraints, however, we will soon populate the huggingface space with proxy images/urls for use.

1. Manually curated and annotated images with intended visual metaphors.
2. Designed to minimize subjectivity and serve as a robust benchmark.


## Instructions for Finetuning

### LLaVA Fine-Tuning
We closely follow the [official LLaVA repository](https://github.com/haotian-liu/LLaVA) for fine-tuning.  
You can use the `collators.py`, `dataset.py`, and `utils.py` modules together with the official implementation to fine-tune LLaVA models efficiently.

### Qwen2-VL Fine-Tuning
We use [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning of [Qwen2-VL](https://colab.research.google.com/drive/1whHb54GNZMrNxIsi2wm2EY_-Pvo2QyKh?usp=sharing).  
You may follow the official implementation closely or alternatively, use `qwen_finetune.py` for your experiments.