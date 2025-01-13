Dataset Information
Oxford Pets Dataset
Description: The Oxford Pets dataset contains images of 37 different pet categories, with approximately 200 images per category. 
The images exhibit significant variations in scale, pose, and lighting conditions. Each image comes with detailed annotations, 
including breed, region of interest (ROI) for the head, and pixel-level foreground-background segmentation (Trimap).
Download URL:https://www.robots.ox.ac.uk/~vgg/data/pets/

Flowers-102 Dataset
Description: The Flowers-102 dataset consists of images of 102 different flower species. 
It is widely used in classification tasks in computer vision and provides richly annotated images for each category.
Download URL:https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
Download from [here](https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view).split_zhou_OxfordFlowers.json

Preparations Before Official Run
Image Filename Specification:
Ensure that the image filenames contain the classification category name. If the filenames do not include the category name,
they must be renamed to include this information. For example, if the images in flower101 lack category names, use the rename.py script to rename them.
Text Description Preparation:
To prepare text descriptions for each image, use the product_text.py script. This will generate two text files: one for the training set and one for the test set. 
Each entry in these files will contain three attributes: file path, text description, and label.
Ablation Study on the Impact of Text Descriptions:
To explore the impact of text descriptions through ablation studies, use the product_sametext.py script. 
This will also generate two text files, but each image will have the same text description.

It is important to note that the DGL must be downloaded from the official website https://www.dgl.ai/pages/start.html, otherwise it is easy to have a version mismatch.
Because this dgl version is easy to mismatch if the official website download fails. can learn from my environment,
Torch：2.3.0+cu121
Python: 3.12;
If the environment is the same as mine, run conda install -c dglteam/label/th23_cu121 dgl=2.4.0.th23.cu121

The final version should look like this:
1.root@autodl-container-75c14f9dbd-39766c8b:~/autodl-tmp/PIXIU# pip show dgl
Name: dgl
Version: 2.4.0+cu121
Summary: Deep Graph Library
Home-page: https://github.com/dmlc/dgl

2.root@autodl-container-75c14f9dbd-39766c8b:~/autodl-tmp/PIXIU# conda list dgl
packages in environment at /root/miniconda3:
Name                    Version                   
dgl             			2.4.0.th23.cu121     


The BLIP model is required to run the product_xxtext.py, and if you want to run it locally, please https://huggingface.co/Salesforce/blip-image-captioning-large/tree/main it in this website. Download these 5 files, config.json, preprocessor_config.json, pytorch_model.bin, tokenizer_config.json, vocab.txt.

More datasets can be found on this website：https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md

Official Run
Model Variants:
linear_base: Represents a simple linear layer model.
linear_ewc: Adds EWC regularization to the linear_base model to mitigate catastrophic forgetting.
Full Model Execution:
pet_run and flower_run scripts are used to run the full model on the respective datasets.
When running the model on different datasets, adjust the num_classes parameter to reflect the number of categories present in the dataset.
To execute the model, modify the train_label_file and test_label_file paths in the configuration to point to the previously generated text files.
