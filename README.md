# Trainer implementation for MixUp and MixMatch algorithms
- **Author:** Andrei Alexandru
- **Faculty:** Faculty of Electronics, Telecommunications and Information Technology
- **Master program:** TAID

This trainer was developed for some ablation studies for my Dissertation project. It supports 3 types of training:
 
 - Normal (ERM mode)
 - With MixUp
 - With MixMatch (Semi-supervised learning)

To train using a certain method:
```bash
python train_emotion.py --method mixup
```
Please check the other arguments in the script for MixMatch and other hyperparams.

If you want to test your model on test images from rafdb use
`test.py` or `test_on_video.py` if you want to test on a video or using your own webcam.
