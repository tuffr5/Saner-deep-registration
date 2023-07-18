# Saner-deep-registration

## TL,DR <a href=self>arXiv</a>
We propose a novel regularization-based sanity-enforcer method that imposes *two sanity checks* on the deep registration model to reduce its inverse consistency errors and increase its discriminative power simultaneously.

## Datasets
<ul>
    <li><a href="https://brain-development.org/ixi-dataset">IXI</a></li>
    <li><a href="https://learn2reg.grand-challenge.org/evaluation/task-3-validation/leaderboard">OASIS</a></li>
    <li><a href="https://www.med.upenn.edu/cbica/brats-reg-challenge"> BraTSReg</a></li>
</ul>
One can always use the google drive data files for IXI and OASIS datasets, kindly processed by Junyu Chen [<a href="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration">here</a>]. A big shout out to his effort.

## Run
If data path is set, simply run sanity_checks_*.py

## Trained Models
Trained models can be found [<a href="https://drive.google.com/drive/folders/1Ph_9T1Iw1YNy_13LgKxPC42mQm0Pxcda?usp=sharing">here</a>].

## Bibtex
```
@inproceedings{duan2023sanity,
  title={Towards Saner Deep Image Registration},
  author={Duan, Bin and Zhong, Ming and Yan, Yan},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
## Acknowledgment
This repo is heavily based on Junyu Chen's and Tony C. W. Mok's codes. Great thanks to them!