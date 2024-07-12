
# A deep learning-based stripe self-correction method for stitched microscopic images

[SSCOR Paper](https://www.biorxiv.org/content/10.1101/2023.01.11.523393v1)

## Pytorch implementation of SSCOR

## OS Requirements
- Linux: Ubuntu 18.04
- Python 3.7 + Pytorch 1.8.1
- NVIDIA GPU + CUDA 11.0 CuDNN 8

## Installation Guide
Install PyTorch and 0.4+ and other dependencies (e.g., torchvision, visdom and dominate).
- For pip users, please type the command pip install -r requirements.txt.
- For Conda users, you can create a new Conda environment using conda env create -f environment.yml.
- To run demo
  - `cd Prestyle` or `cd PostGener`

## Training Image Data
Before training, the large image was divided into 256*256 pixels.Uncomment in the fenge file to set labels for multi-style images.
```bash
python fenge.py
```
The SourceA image train-data are placed in `./datasets/data/trainA` directory.
The SourceB image train-data are placed in `./datasets/data/trainB` directory.
The SourceA image test-data are placed in `./datasets/data/testA` directory.
The SourceB image test-data are placed in `./datasets/data/testB` directory.
The test result are placed in `./datasets/data/testC` directory.

## PreStyleNet train/test
```bash
python train.py --dataroot ./datasets/data --name name --cls --lab --input_nc 4 --output_nc 4
python test.py --dataroot ./datasets/data --name name --cls --hebing --chutu ./datasets/data/testC --input_nc 4 --output_nc 4 --lab --shuchu 'fake_B'
```

## PostGenerNet train/test
```bash
python train.py --dataroot ./datasets/data --name name --Auxiliary 1
python test.py --dataroot ./datasets/data --name name --cls --hebing --chutu ./datasets/data/testC --shuchu 'fake_B'
```
To see more intermediate results, check out ./checkpoints/. The .pth file will be save in the corresponding folder.

## Contact
If you have any questions, please contact Shu Wang at [shu@fzu.edu.cn](shu@fzu.edu.cn) or Wenxi Liu at [wenxiliu@fzu.edu.cn](wenxiliu@fzu.edu.cn).
