This is the repository for the final project in DTU course 02456 Deep Learning by group 76. 



**Notebooks**

Notebooks for recreating our results can be found [here](Model/Notebooks). 

The semi-supervised models have been trained on DTU's HPC cluster on thus load our [pre-trained models](Trained_models), to produce results. 

Baseline U-Net with and without data augmentation is combined in the same [notebook](Model/Notebooks/Baseline_UNet.ipynb). They can be run and trained within reasonable time on CPU, to produce results. The notebook was developed and run in Colab, and we were unable to upload the final model .pth files to the Github owing to file size. For this reason we recommend running it in colab.


**Scripts**

Scripts have been used for all semi-supervised models, to perform the training on the HPC. Scripts can be found [here](Model). 
