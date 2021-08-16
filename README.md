# AuxSegNet

The pytorch code for our ICCV 2021 paper [Leveraging Auxiliary Tasks with Affinity Learning for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2107.11787).

#### Train the AuxSegNet

```
python train_AuxAff.py --img_path 'Path to the training images'\
                       --seg_pgt_path 'Path to the pseudo segmentation labels' \
                       --sal_pgt_path 'Path to the pseudo saliency labels' \
                       --init_weights 'Path to the initialization weights' \
                       --save_path 'Path to save the trained AuxSegNet model' 
```


#### Pseudo label updating
```
python gen_pgt.py --weights 'path to the trained AuxSegNet weights'\   
                  --img_path 'Path to the training images'\
                  --SALpath 'Path to the pre-trained saliency maps' \
                  --seg_pgt_path 'Path to save updated pseudo segmentation labels' \
                  --sal_pgt_path 'Path to save updated pseudo saliency labels' 
```


#### Inference
```
python infer_AuxAff.py --img_path 'Path to the training images'\
                       --weights 'Path to the trained AuxSegNet weights'\
                       --save_path 'Path to save the segmentation results'