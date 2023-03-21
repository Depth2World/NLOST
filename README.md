NLOST-Non-Line-of-Sight-Imaging-with-Transformer
===

## Reconstructed hidden scenes from the public real-world dataset.
![fk_rw](https://github.com/Depth2World/NLOST/blob/main/images/fk_rw.png)

## Reconstructed hidden scenes from real-world measurements captured by our NLOS imaging system.
![ours_rw](https://github.com/Depth2World/NLOST/blob/main/images/our_rw.png)
## More details about the imaging system.
We built a NLOS system working in a confocal manner. A 532 nm laser emits pulses at 50 ps pulse width and 11 MHz repetition frequency with a typical 250 mW average power. The pulses pass through a two-axis raster-scanning galvo mirror, and transmit to the visible wall. The direct and indirect diffuse photons are collected by the other two-axis galvo mirror and then coupled to a multimode fiber directed to a free-running single-photon avalanche diode (SPAD) detector with a detection efficiency about 40\%. A time-correlated single photon counter records the sync signals from the laser and the photon-detection signals from the SPAD. The temporal resolution of the overall system is measured to be approximately 95 ps. 

In data collection, the illuminated point and sampling point keep the same direction but slightly misaligned (to avoid the first bouncing signals) during scanning. We raster scan the square grid of points across a 2m * 2m area on the visible wall. The acquisition time for each scanning point is set to be about 8 ms. The histogram length of each transient measurement is 512, with a bin width of 32 ps. The NLOS scenes include a ladder with letters, sculptures of people and deer made of white foam, which is placed about 0.8 m to 1.5 m away from the visible wall. 

The scene thumbnails of the captured measurements are shown as below.
![scene thumbnails](https://github.com/Depth2World/NLOST/blob/main/images/objects.png)
The measurements can be downloaded at [googledisk]().

## Contact 
For questionss, feel free to contact YueLi (yueli65@mail.ustc.edu.cn) or Jiayong Peng (jiayong@mail.ustc.edu.cn).

## Citation

## Acknowledgements
We thank the authors that shared the code of their works. In particular:
* Wenzhen Chen [LFE](https://github.com/princeton-computational-imaging/NLOSFeatureEmbeddings)
* Siyuan Shen [NeTF](https://github.com/zeromakerplus/NeTF_public)
