# AuxSegNet

The pytorch code for our ICCV 2021 paper [Leveraging Auxiliary Tasks with Affinity Learning for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2107.11787).

<p align="left">
  <img src="mis/framework2.jpg" width="auto" title="" >
</p>

#### 1. Prepare initial pseudo labels
- Off-the-shelf saliency maps used as the initial saliency pseudo labels. [[DSS]](https://drive.google.com/open?id=1Ls2HBtg3jUiuk3WUuMtdUOVUFCgvE8IX)
- Extract the class activation maps (CAM) from a pre-trained single-task classification network. [[ResNet38]](https://drive.google.com/file/d/1xESB7017zlZHqxEWuh1Rb89UhjTGIKOA/view?usp=sharing)
- Generate the initial pseudo segmentation labels using the above saliency and CAM maps via [[heuristic fusion]](https://github.com/xulianuwa/AuxSegNet/blob/597506a4f44cca81d11c986217e5318361e8f65e/tool/imutils.py#L36).
#### 2. Train the AuxSegNet

```
python train_AuxAff.py --img_path 'Path to the training images'\
                       --seg_pgt_path 'Path to the pseudo segmentation labels' \
                       --sal_pgt_path 'Path to the pseudo saliency labels' \
                       --init_weights 'Path to the initialization weights' \
                       --save_path 'Path to save the trained AuxSegNet model' 
```


#### 3. Pseudo label updating
```
python gen_pgt.py --weights 'path to the trained AuxSegNet weights'\   
                  --img_path 'Path to the training images'\
                  --SALpath 'Path to the pre-trained saliency maps' \
                  --seg_pgt_path 'Path to save updated pseudo segmentation labels' \
                  --sal_pgt_path 'Path to save updated pseudo saliency labels' 
```
#### 4. Iterate Step 2 and 3

#### 5. Inference
```
python infer_AuxAff.py --img_path 'Path to the training images'\
                       --weights 'Path to the trained AuxSegNet weights'\
                       --save_path 'Path to save the segmentation results'

```

## Citation
Please cite our paper if the code is helpful.
```
@inproceedings{xu2021leveraging,
  title={Leveraging Auxiliary Tasks with Affinity Learning for Weakly Supervised Semantic Segmentation},
  author={Xu, Lian and Ouyang, Wanli and Bennamoun, Mohammed and Boussaid, Farid and Sohel, Ferdous and Xu, Dan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
