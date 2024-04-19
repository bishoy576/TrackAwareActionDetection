# Track Aware Action Detection (TAAD)

Here you find the evaluation and training for [TAAD](https://arxiv.org/abs/2209.02250) on [MultiSports](https://deeperaction.github.io/datasets/multisports.html) Dataset. TAAD finished in the top spot in [MultiSport challenge](https://deeperaction.github.io/results/index.html) held in conjunction with ECCV 2022.

## License

This repo heavily relies on [PySlowFast](https://github.com/facebookresearch/SlowFast) so it contains a lot of stuff from there. It has the same License as PySlowFast, which is released under the [Apache 2.0 license](LICENSE).

## Model Zoo and Baselines

Multisport TAAD TCN model and multisports tracks generteted using Yolov5 and DeepSORT are availble to Download from [Googl-Drive](https://drive.google.com/drive/folders/1KP6asw4vZb-TuD26DLMoTAwQzczmNQQQ?usp=sharing).

## Installation

Please find installation instructions for PyTorch and PySlowFast in [INSTALL.md](INSTALL.md).

## Dataset Preparation
Download Track from [Googl-Drive](https://drive.google.com/drive/folders/1KP6asw4vZb-TuD26DLMoTAwQzczmNQQQ?usp=sharing). and Extract frame using multisport script provided by authors of Mulitsports.

## Eval
Given the extracted frames and downloaed tracks and TAAD-TCN model from [Googl-Drive](https://drive.google.com/drive/folders/1KP6asw4vZb-TuD26DLMoTAwQzczmNQQQ?usp=sharing). Now, you shoiuld be able to run TAAD-TCN which achives the best perfromance on Mulitsports.  

## Train
Unfortunalty I do not have the bandwidth to reproduce this myself. But it should be doable with given current set of code base. 

## Contributors
TAAD is written by [Gurkirt Singh](https://gurkirt.github.io/). I am happy to recieve a pull request for training reproduction. 

## Citing PySlowFast
If you find this useful in your research, please use the following BibTeX entry for citation.
```BibTeX
@InProceedings{Singh_2023_WACV,
    author    = {Singh, Gurkirt and Choutas, Vasileios and Saha, Suman and Yu, Fisher and Van Gool, Luc},
    title     = {Spatio-Temporal Action Detection Under Large Motion},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {6009-6018}
}
```
Don't forget to cite orignal PySlowfast Repo.


```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```
