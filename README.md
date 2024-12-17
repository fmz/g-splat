
# G-Splat: Gaussian Splatting, but written for CS1430 final project (:

Adopted from [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
By Kerbl et. al

This is a final project code submission for Brown University's [CS1430](https://browncsci1430.github.io)
With Professor Srinath Sridhar

## Running the Code
* [Setup](#setup)
* [Running the Code](#demo)

### Setup
The code has been tested in the following setup
* Ubuntu 22.04 and WSL/Ubuntu24.04
* Python 3.12
* PyTorch 2.5
* CUDA 12.4-12.6

The provided environment.yml serves as a suggestion:
```
$ conda env create -f environment.yml
$ conda activate g-splat
```

You need to compile the `diff-gaussian-rasterizer` and `fused-ssim` manually:

NOTE: You might run into a compilation issue in `diff-gaussian-rasterizer`. Easily fixable by adding `#include <cstdint>` in the file that complains.
```
$ pip install 3rdparty/diff-gaussian-rasterization
$ pip install 3rdparty/fused-ssim
```

### Running the Code
We have 3 modes:

1. Single-image: overfits on a single image
2. Blender: Runs on our synthetic Blender-generated datasets
3. Colmap: More in-line with the upstream gaussian splatting data format

See examples below

#### Overfitting a single image
```
python main.py --mode blender --data data/yosemite1.jpg --viz_interval 100
```
#### Overfitting Blender data
```
python main.py --mode blender --data data/monkey --viz_interval 5
```
#### Overfitting Colmap data
```
python main.py --mode colmap --data data/db/drjohnson --viz_interval 5
```

### Notes
The code is pretty bare-bones. Please adjust they hyperparameters stright in the
code. Specifically see main.py and scene.py.

## License
Not for commercial use :)