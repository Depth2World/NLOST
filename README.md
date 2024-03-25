## NLOST-Non-Line-of-Sight-Imaging-with-Transformer 
Paper: [CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_NLOST_Non-Line-of-Sight_Imaging_With_Transformer_CVPR_2023_paper.pdf)

### $\color{red}{New!}$ We provide two versions trained in size 256 * 256 * 512 and size 128 * 128 * 512. The pre-trained models are available now. The inference memory is about 22G~23G.

### Reconstructed hidden scenes from the real-world measurements captured by [FK](https://github.com/computational-imaging/nlos-fk).
![fk_rw](https://github.com/Depth2World/NLOST/blob/main/images/fk_rw.png)


### Reconstructed hidden scenes from the real-world measurements captured by our NLOS imaging system.
![ours_rw](https://github.com/Depth2World/NLOST/blob/main/images/our_rw.png)


### Details about our self-built imaging system.
We built an NLOS system working in a confocal manner. A 532 nm laser emits pulses at 50 ps pulse width and 11 MHz repetition frequency with a typical 250 mW average power. The pulses pass through a two-axis raster-scanning galvo mirror, and transmit to the visible wall. The direct and indirect diffuse photons are collected by the other two-axis galvo mirror and then coupled to a multimode fiber directed to a free-running single-photon avalanche diode (SPAD) detector with a detection efficiency about 40\%. A time-correlated single photon counter records the sync signals from the laser and the photon-detection signals from the SPAD. The temporal resolution of the overall system is measured to be approximately 95 ps. 

In data collection, the illuminated point and sampling point keep the same direction but are slightly misaligned (to avoid the first bouncing signals) during scanning. We raster scan the square grid of points across a 2m * 2m area on the visible wall. The acquisition time for each scanning point is set to be about 8 ms. The histogram length of each transient measurement is 512, with a bin width of 32 ps. 

### Datasets
The NLOS scenes include a ladder with letters on the ladder, sculptures of people, and deer made of white foam, which are placed about 0.8 m to 1.5 m away from the visible wall. 
The scene thumbnails of the captured measurements are shown below.
![scene thumbnails](https://github.com/Depth2World/NLOST/blob/main/images/objects.png)

The real-world measurements captured by our imaging system can be downloaded at [googledisk](https://drive.google.com/file/d/1ybJxH5nBr7Fv3heSl5BRP70nuBCfMYc5/view?usp=sharing).

### Training
1. For evaluation on the real-world data, we trained the models on the synthetic data (~3000 motorbike dataset) provided by [LFE](https://github.com/princeton-computational-imaging/NLOSFeatureEmbeddings). \
You can download [Here](https://drive.google.com/file/d/183VAD_wuVtwkyvfaBoguUHZgHu065BNW/view)


2. We also utilized the real-world data provided by [FK](https://github.com/computational-imaging/nlos-fk).\
You can download the preprocessed data [Here](https://drive.google.com/file/d/1Zf4BAwzkEesltx7zUJvWcvxa1n4XH4U2/view?usp=sharing).

3. run ' bash train.sh'

### Test
1. The pre-trained models are listed in './pretain'

2. run 'bash test.sh'

3. For **Unseen** data, you can download [Here](https://drive.google.com/file/d/1gsFfSnZ07QOCHrxDlOnRtZiIFhqvjVAc/view?usp=sharing).
   

### Contact 
For questions, feel free to contact YueLi (yueli65@mail.ustc.edu.cn).

### Acknowledgements
We thank the authors who shared the code of their works. Particularly
  Wenzhen Chen [LFE](https://github.com/princeton-computational-imaging/NLOSFeatureEmbeddings) and 
  Fangzhou Mu [PTR](https://github.com/fmu2/nlos3d).
### Citation
If you find it useful, please cite our paper.

    @inproceedings{li2023nlost,
      title={NLOST: Non-Line-of-Sight Imaging with Transformer},
      author={Li, Yue and Peng, Jiayong and Ye, Juntian and Zhang, Yueyi and Xu, Feihu and Xiong, Zhiwei},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={13313--13322},
      year={2023}
    }
