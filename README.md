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

Official Run
Model Variants:
linear_base: Represents a simple linear layer model.
linear_ewc: Adds EWC regularization to the linear_base model to mitigate catastrophic forgetting.
Full Model Execution:
pet_run and flower_run scripts are used to run the full model on the respective datasets.
When running the model on different datasets, adjust the num_classes parameter to reflect the number of categories present in the dataset.
To execute the model, modify the train_label_file and test_label_file paths in the configuration to point to the previously generated text files.
