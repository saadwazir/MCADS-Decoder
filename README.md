# MCADS-Decoder

**Rethinking Decoder Design:<br>Improving Biomarker Segmentation Using Depth-to-Space Restoration and Residual Linear Attention**
***- Accepted in CVPR 2025***


Download Paper:
https://openaccess.thecvf.com/content/CVPR2025/html/Wazir_Rethinking_Decoder_Design_Improving_Biomarker_Segmentation_Using_Depth-to-Space_Restoration_and_CVPR_2025_paper.html

Please Cite it as following

```
@inproceedings{wazir2025rethinking,
  title={Rethinking decoder design: Improving biomarker segmentation using depth-to-space restoration and residual linear attention},
  author={Wazir, Saad and Kim, Daeyoung},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={30861--30871},
  year={2025},
  doi = {10.48550/arXiv.2506.18335},
  url = {https://doi.org/10.48550/arXiv.2506.18335}
}
```
_____________________________________________________________________________

<img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExNTUwN2E2czJlZjdsOWZwZnkwdnVoZTBzbHNvOHk1cml6bGM0NXF4bSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/O7BtZQ0ceoVz4pDdC7/giphy.gif" width="100" /> Experimental Results* on the [MedCAGD-Dataset-Collection](https://huggingface.co/datasets/saadwazir/MedCAGD-Dataset-Collection)

Download Dataset from Huggingface. Link: https://huggingface.co/datasets/saadwazir/MedCAGD-Dataset-Collection

<table style="width:100%; border-collapse: collapse; text-align: center;" border="1">
  <caption style="font-weight: bold; margin-bottom: 8px;">TABLE 1: ACDC DATASET RESULTS (MULTI-CLASS SEMANTIC SEGMENTATION TASK)</caption>
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th>Method</th>
      <th>Dice ↑</th>
      <th>IoU ↑</th>
      <th>HD95 ↓</th>
      <th>RV</th>
      <th>Myo</th>
      <th>LV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>U-Net</td>
      <td>81.56</td>
      <td>73.41</td>
      <td>6.9854</td>
      <td>76.99</td>
      <td>80.28</td>
      <td>87.43</td>
    </tr>
    <tr>
      <td><strong>MCADS</strong></td>
      <td><strong>84.51</strong></td>
      <td><strong>76.92</strong></td>
      <td><strong>5.5595</strong></td>
      <td><strong>81.16</strong></td>
      <td><strong>83.27</strong></td>
      <td><strong>89.09</strong></td>
    </tr>
  </tbody>
</table>

<table style="width:100%; border-collapse: collapse; text-align: center;" border="1">
  <caption style="font-weight: bold; margin-bottom: 8px;">TABLE 2: SYNAPSE DATASET RESULTS (MULTI-CLASS SEMANTIC SEGMENTATION TASK)</caption>
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th>Method</th>
      <th>Dice ↑</th>
      <th>IoU ↑</th>
      <th>HD95 ↓</th>
      <th>Aorta</th>
      <th>GB</th>
      <th>KL</th>
      <th>KR</th>
      <th>Liver</th>
      <th>PC</th>
      <th>SP</th>
      <th>SM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>U-Net</td>
      <td>70.11</td>
      <td>59.39</td>
      <td>44.69</td>
      <td>84.00</td>
      <td>56.70</td>
      <td>72.41</td>
      <td>62.64</td>
      <td>86.98</td>
      <td>48.73</td>
      <td>81.48</td>
      <td>67.96</td>
    </tr>
    <tr>
      <td><strong>MCADS</strong></td>
      <td><u>85.03</u></td>
      <td><u>81.71</u></td>
      <td><strong>11.11</strong></td>
      <td>90.81</td>
      <td><u>86.07</u></td>
      <td>86.77</td>
      <td>83.24</td>
      <td>87.66</td>
      <td><strong>83.55</strong></td>
      <td>85.74</td>
      <td>76.38</td>
    </tr>
    <tr>
      <td>Self-Prompt SAM</td>
      <td><strong>86.74</strong></td>
      <td>-</td>
      <td>-</td>
      <td><strong>91.99</strong></td>
      <td>69.95</td>
      <td><strong>85.65</strong></td>
      <td><strong>85.40</strong></td>
      <td><strong>97.39</strong></td>
      <td>79.18</td>
      <td><strong>94.38</strong></td>
      <td><strong>89.94</strong></td>
    </tr>
  </tbody>
</table>


<table style="width:100%; border-collapse: collapse; text-align: center;" border="1">
  <caption style="font-weight: bold; margin-bottom: 8px;">TABLE 3: RESULTS ON MULTIPLE DATASETS (BINARY SEMANTIC SEGMENTATION TASK)</caption>
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th rowspan="2">Method</th>
      <th rowspan="2">Params ↓</th>
      <th rowspan="2">Flops ↓</th>
      <th colspan="2">Skin</th>
      <th colspan="2">Polyp</th>
      <th colspan="2">Fundus</th>
      <th colspan="2">Neoplasm</th>
      <th>Cell</th>
      <th>All</th>
    </tr>
    <tr style="background-color: #f2f2f2;">
      <th>ISIC17</th>
      <th>ISIC18</th>
      <th>ETIS</th>
      <th>ColonDB</th>
      <th>DRIVE</th>
      <th>FIVES</th>
      <th>BUSI</th>
      <th>ThyroidXL</th>
      <th>CellSeg</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>U-Net</td>
      <td>34.53 M</td>
      <td>65.53 G</td>
      <td>83.07</td>
      <td>86.67</td>
      <td>76.85</td>
      <td>83.95</td>
      <td>71.20</td>
      <td>75.77</td>
      <td>74.04</td>
      <td>71.16</td>
      <td>71.52</td>
      <td>77.14</td>
    </tr>
    <tr>
      <td><strong>MCADS</strong></td>
      <td>50.90 M</td>
      <td>61.89 G</td>
      <td><strong>84.14</strong></td>
      <td><u><strong>91.01</strong></u></td>
      <td><u><strong>92.24</strong></u></td>
      <td><strong>91.37</strong></td>
      <td><u><strong>78.42</strong></u></td>
      <td><strong>76.05</strong></td>
      <td><strong>80.03</strong></td>
      <td><u><strong>86.33</strong></u></td>
      <td><strong>86.68</strong></td>
      <td><u>85.14</u></td>
    </tr>
    <tr>
      <td>AutoSam</td>
      <td>41.56 M</td>
      <td>25.11 G</td>
      <td>-</td>
      <td>-</td>
      <td>79.70</td>
      <td>83.00</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Medical SAM3</td>
      <td>840.0 M</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>86.10</td>
      <td>-</td>
      <td>55.80</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

Research Note *
This dataset collection provides early access to the datasets used for benchmarking segmentation models across multiple medical imaging datasets. The segmentation benchmarks associated with this dataset collection are part of ongoing research related to the MCADS decoder and the upcoming MedCAGD framework. The full benchmark results and evaluation protocols will appear in the MedCAGD paper, which is currently under review, and additional results will be released after the review process.
_____________________________________________________________________________
### Setup Conda Environment
use this command to create a conda environment (all the required packages are listed in `mcadsDecoder_env.yml` file)
```
conda env create -f mcadsDecoder_env.yml
```





### Datasets

#### MoNuSeg - Multi-organ nuclei segmentation from H&E stained histopathological images.
link: https://monuseg.grand-challenge.org/Data/

#### TNBC - Triple-negative breast cancer.
link: https://zenodo.org/records/1175282#.YMisCTZKgow

#### DSB - 2018 Data Science Bowl.
link: https://www.kaggle.com/c/data-science-bowl-2018/data

#### EM - Electron Microscopy.
link: https://www.epfl.ch/labs/cvlab/data/data-em/

### Data Preprocessing
After downloading the dataset you must generate patches of images and their corresponding masks (Ground Truth), & convert it into numpy arrays or you can use dataloaders directly inside the code. Note: The last channel of masks must have black and white (0,1) values not greyscale(0 to 255) values. 
you can generate patches using Image_Patchyfy. Link : https://github.com/saadwazir/Image_Patchyfy

### Offline Data Augmentation
(it requires albumentations library link: https://albumentations.ai)

use `offline_augmentation.py` to generate augmented samples




## Training and Testing

1. Edit the `config.txt` file to set training and testing parameters and define folder paths.
2. Run the `mcadsDecoder.py` file in a conda environment. It contains the model, training, and testing code.





---

## Configurations

- Paths for training
  
Define paths for folders that contain patches of images and masks for training.

```
train_images_patch_dir=/mnt/hdd_2A/datasets/monuseg_patches_augm/images/
train_masks_patch_dir=/mnt/hdd_2A/datasets/monuseg_patches_augm/masks/
```

- Paths for testing
  
Define paths for numpy arrays that contain patches of images and masks for testing.

```
test_images_patch_dir=/mnt/hdd_2A/datasets/monuseg_test_patches_arrays/monuseg_org_X_test.npy
test_masks_patch_dir=/mnt/hdd_2A/datasets/monuseg_test_patches_arrays/monuseg_org_y_test.npy
```

Define paths for folders that contain full-size images and masks for testing.

```
image_full_test_directory=/mnt/hdd_2A/datasets/monuseg_org/test/image/
mask_full_test_directory=/mnt/hdd_2A/datasets/monuseg_org/test/mask/
```

- Training Parameters
```
training=False
gpu_device=0
num_epochs=200
batch_size=8
imgz_size=256
```

- Evaluation Parameters
  
Parameters for processing patches of images and masks:
  
```
patch_img_size=256
patch_step_size=128
```
```
resize_img=True #set resize_img=False if full image sizes have different width and height.
resize_height_width=1024
```

Parameters for processing full-size images and masks:
  
```
resize_full_images=True #if resize_full_images=False then full-size images are not scaled down, but evaluation takes more time.
```


