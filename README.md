# Albedo Recovery for Aerial Photogrammetric Images

This repository contains the upcoming implementation of the albedo recovery approach described in the paper: [A general albedo recovery approach for aerial photogrammetric images through inverse rendering](https://authors.elsevier.com/a/1jlhp3I9x1qnxt). 

This work extends our prior work [“A Novel Intrinsic Image Decomposition Method to Recover Albedo for Aerial Images in Photogrammetry Processing”](https://isprs-annals.copernicus.org/articles/V-2-2022/23/2022/) presented at the ISPRS Congress 2022.

<video src="https://github.com/user-attachments/assets/f0972c11-e748-4532-a57a-ee41de9256fc" width="100%" controls autoplay loop></video>
<video src="https://github.com/user-attachments/assets/146038c2-effa-422a-aa1a-5f7ad17f5830" width="100%" controls autoplay loop></video>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
<tr>
<td align='center'>Original Texture</td>
<td align='center'>Agisoft Delighter</td>
<td align='center'>Ours</td>
</tr>
<tr>
<td>
<video src="https://github.com/user-attachments/assets/a37893e2-3d02-4500-9a91-e8d0750d1198" width="100%" controls autoplay loop></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/eababd5b-e9e2-4a19-8d1a-e8833c8efc5c" width="100%" controls autoplay loop></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/b4841e14-5cd3-4be0-bc75-d371a5152e6b" width="100%" controls autoplay loop></video>
</td>
</tr>
</table>

Thank you for your interest in our work. We look forward to sharing our tools and seeing the amazing results you achieve with them!

## Code Release Roadmap

We are actively working on preparing the code for public release. Below is our planned release roadmap:

| Milestone | Planned Date | Status |
|----|----| ----|
|Synthetic dataset release | 2024/11 | Released |
|Run Code release | 2024/12 | Released |
|Evaluation and Experiments | 2024/12 | Pending |

## Synthetic Dataset
For detailed information about the synthetic dataset, please refer to the [Synthetic Datasets Documentation](./doc/synthetic_datasets.md).

## Read-world Dataset Preparation Guideline
Please refer to [Real-world Dataset Preparation Guideline](./doc/dataset_preprocessing.md).

## Citations

##### ISPRS Journal 2024:

``` bibtex
@article{SONG2024101,
title = {A general albedo recovery approach for aerial photogrammetric images through inverse rendering},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {218},
pages = {101-119},
year = {2024},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2024.09.001},
url = {https://www.sciencedirect.com/science/article/pii/S0924271624003319},
author = {Shuang Song and Rongjun Qin},
keywords = {Inverse rendering, Albedo recovery, Aerial photogrammetry, Shading, Ray-tracing, Dense matching},
}
```

##### ISPRS Congress Nice 2022:

``` bibtex
@Article{isprs-annals-V-2-2022-23-2022,
AUTHOR = {Song, S. and Qin, R.},
TITLE = {A NOVEL INTRINSIC IMAGE DECOMPOSITION METHOD TO RECOVER ALBEDO FOR AERIAL IMAGES IN PHOTOGRAMMETRY PROCESSING},
JOURNAL = {ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
VOLUME = {V-2-2022},
YEAR = {2022},
PAGES = {23--30},
URL = {https://isprs-annals.copernicus.org/articles/V-2-2022/23/2022/},
DOI = {10.5194/isprs-annals-V-2-2022-23-2022}
}
```

## Contact

Shuang Song: song.1634@osu.edu
Rongjun Qin: qin.324@osu.edu

