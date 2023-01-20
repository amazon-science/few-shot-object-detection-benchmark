
## Preparing Datasets

### Folder structure
In general, we need to create datasets following the structures below

    datasets
    │   README.md
    └───iwildcam
    |   |   conversion.py
    │   └───JPEGImages
    |       |
    |       └───animals
    │           │   ...
    |   |
    |   └───OI_Annotations
    |       |   train.csv
    |       |   test.csv
    |       |   class_map.csv
    │   
    └───oktoberfest
    │   │   README.MD
    |   |   conversion.py
    │   └───JPEGImages
    |   └───OI_Annotations
    |   ...


Under the master `datasets` folder, we have 10 sub-folders, corresponding to each dataset
in the MoFSOD benchmark. Under each dataset folder (e.g., `iwildcam`), we have two sub-folders:
`JPEGImages`, which stores all the images, and `OI_Annotations`, the OpenImages style annotation files and class map in CSV format.
The annotations are named: `test.csv` `train.csv` and `class_map.csv`.

Inside each dataset folder, we also have a `conversion.py` file to convert original annotation of the dataset to desired OpenImages format.
For your convenience, we provided all the converted annotations:

    

However, you will still need to setup the images folder as instructed.

### 1. iWildCam

iWildCam's images can be downloaded from the github page. 

    https://github.com/visipedia/iwildcam_comp/tree/master/2020
    https://lila.science/datasets/wcscameratraps

Extract the downloaded images to `JPEGImages`. Each image should follow the format `JPEGImages/animals/xxxx/xxxx.jpg`.

Note that we use the 2020 challenge annotations, which currently is available here:

    https://lilablobssc.blob.core.windows.net/wcs/wcs_20200403_bboxes.json.zip

After the preparation, run the conversion script. This should generate `test.csv` `train.csv` and `class_map.csv` in the `OI_Annotations` folder.

    python conversion.py --train_json_file wcs_20200403_bboxes.json --output_dir ./OI_Annotations/


### 2. VisDrone2019

VisDrone2019 dataset can be downloaded from the github page

    https://github.com/VisDrone/VisDrone-Dataset

Note that we use only the image detection annotations. After downloading and extracting the images and annotations to the `visdrone2019` folder,
there should be 3 sub-folders: `VisDrone2019-DET-train`, `VisDrone2019-DET-val` and `VisDrone2019-DET-test-dev`. Each sub-folder should contain
`annotations` and `images`. 

Then, create symbolic link for each `images` folder to the `JPEGImages` with:

    ln -s VisDrone2019-DET-train/images JPEGImages/train2019
    ln -s VisDrone2019-DET-val/images JPEGImages/val2019
    ln -s VisDrone2019-DET-test-dev/images JPEGImages/test2019

After the preparation, run the conversion script. This should generate `test.csv` `train.csv` and `class_map.csv` in the `OI_Annotations` folder.

    python conversion.py --root (your absolute path to the visdrone2019 folder)


### 3. Oktoberfest

Oktoberfest dataset can be downloaded from the github page

    https://github.com/a1302z/OktoberfestFoodDataset

After downloading and extracting the images and annotations to the `oktoberfest` folder,
there should be 2 sub-folders containing the images: `train` and `test`, as well as two annotations files: `train_files.txt` and `test_files.txt`.

Then, create symbolic link for each folder to the `JPEGImages` with:

    ln -s train JPEGImages/train
    ln -s test JPEGImages/test

After the preparation, run the conversion script. This should generate `test.csv` `train.csv` and `class_map.csv` in the `OI_Annotations` folder.

    python conversion.py --root (your absolute path to the oktoberfest folder)


### 4. Clipart

Clipart dataset can be downloaded from the github page

    https://github.com/naoto0804/cross-domain-detection/tree/master/datasets

Only ClipArt1K is needed. After downloading and extracting the images and annotations to the `clipart` folder,
there should be 3 sub-folders containing the images: `JPEGImages`, annotations: `Annotations` and splits: `ImageSets/Main`.

After the preparation, run the conversion script. This should generate `test.csv` `train.csv` and `class_map.csv` in the `OI_Annotations` folder.

    python conversion.py --root (your absolute path to the clipart folder)


### 5. Fashionpedia

Fashionpedia dataset can be downloaded from the github page

    https://github.com/cvdfoundation/fashionpedia#images

Note that we use only the image detection annotations, i.e. `instances_attributes_train2020` and `instances_attributes_val2020`. After downloading 
and extracting the annotations to the `fashionpedia` folder,  and the images to the JPEGImages folder as `train` and `val`, respectively.

After the preparation, run the conversion script. This should generate `test.csv` `train.csv` and `class_map.csv` in the `OI_Annotations` folder.

    python conversion.py --train_json_file (your downloaded instances_attributes_train2020.json file from above)
    --test_json_file (your downloaded instances_attributes_val2020.json file from above) --output_dir ./OI_Annotations/

### 6. LogoDet-3K

LogoDet-3K dataset can be downloaded from the github page

    https://github.com/Wangjing1551/LogoDet-3K-Dataset

After downloading the dataset and extracting the annotations to the `logodet_3k` folder, there should be 2 folders: `LogoDet-3K` and `annotations`.
Then run:

    ln -s LogoDet-3K JPEGImages

After the preparation, run the conversion script. This should generate `test.csv` `train.csv` and `class_map.csv` in the `OI_Annotations` folder.

    python conversion.py --train_json_file annotations/train.json --test_json_file annotations/val.json --output_dir ./OI_Annotations/


### 7. KITTI

KITTI dataset can be downloaded from the link below. Note that we use this version:

    http://www.svcl.ucsd.edu/projects/universal-detection/

After downloading the dataset and extracting the annotations to the `kitti` folder, there should be 3 sub-folders 
containing the images: `JPEGImages`, annotations: `Annotations` and splits: `ImageSets/Main`.

After the preparation, run the conversion script. This should generate `test.csv` `train.csv` and `class_map.csv` in the `OI_Annotations` folder.

    python conversion.py --root (your absolute path to the clipart folder)


### 8. DeepFruits

DeepFruits dataset can be downloaded from the author's personal website. Note that the website is currently down. 
Temporarily the dataset could be downloaded here:

    https://drive.google.com/file/d/10o-_UAlEgGqeWM4gKgyFMwanAC2dV2vq

After downloading the dataset and extracting the annotations to the `deepfruits` folder, there should be 3 sub-folders 
containing the images: `datasets`, annotations: `annotations`. There should a `train.json` and a `test.json` in `annotations` folder. 
Run the following command:

    ln -s datasets JPEGImages

After the preparation, run the conversion script. This should generate `test.csv` `train.csv` and `class_map.csv` in the `OI_Annotations` folder.

    python conversion.py --train_json_file annotations/train.json --test_json_file annotations/test.json --output_dir ./OI_Annotations/


### 9. CrowdHuman

CrowdHuman dataset can be downloaded from the here:

    https://www.crowdhuman.org/

After downloading the dataset and extracting the annotations to the `crowdhuman` folder, there should be 3 sub-folders 
containing the images: `Images`, annotations: `Annotation` and  `splits`.

After the preparation, run the conversion script. This should generate `test.csv` `train.csv` and `class_map.csv` in the `OI_Annotations` folder.

    python convert_coco.py --train_file annotations/annotation_train.odgt --test_file annotations/annotation_val.odgt --output_dir ./annotations/
    python conversion.py --train_json_file annotations/train.json --test_json_file annotations/test.json --output_dir ./OI_Annotations/

### 9. SIXray

SIXray dataset can be downloaded from the github page:

    https://github.com/MeioJane/SIXray

As this datasets contains multiple training and testing splits. To make things easy, we directly provide the converted OI_Annotations files.


Please cite our paper as well as all the datasets used in the benchmark. 



