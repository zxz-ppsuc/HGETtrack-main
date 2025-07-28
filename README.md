# HGETtrack
The tracking results for HGETtrack have been released.

- [HOTC](https://www.hsitracking.com/) ([results]https://github.com/zxz-ppsuc/HGETtrack-main/blob/main/RGBT_workspace/results/challenge2023/result1/)
--------------------------------------------------------------------------------------


##  Install
```
git clone https://github.com/zxz-ppsuc/HGETtrack-main.git
```
## Environment
 > * CUDA 11.8
 > * Python 3.9.18
 > * PyTorch 2.0.0
 > * Torchvision 0.15.0
 > * numpy 1.25.0 
 - Please check the `requirement.txt` for details.

## Usage
- Download the Hyperspectral training/test datasets: [HOTC](https://www.hsitracking.com/).
- Download the pretrained model: [pretrained model](https://pan.baidu.com/s/1n95fom7Fe0bJuEB_GfTgNw?pwd=p9ai) (code: p9ai) to `./pretrained_models/`.
- Please train the HGETtrack based on the [foundation model](https://github.com/botaoye/OSTrack) .
- We will release the well-trained model of [HGETtrack](https://pan.baidu.com/s/1aNBCHMeggcB-N8RCqU9wdQ?pwd=3mpt) (code: 3mpt).
- The generated model will be saved to the path of `./output/checkpoints/train/vipt`.
- Please test the model. The results will be saved in the path of `./RGBT_workspace/results/challenge2023/vitb_384_mae_ce_32x4_got10k_ep100`.
- For evaluation, please download the evaluation benchmark [Toolkit](http://cvlab.hanyang.ac.kr/tracker_benchmark/) and [vlfeat](http://www.vlfeat.org/index.html) for more precision performance evaluation.
- Refer to [HOTC](https://www.hsitracking.com/hot2022/) for evaluation.
- Evaluation of the HGETtrack tracker. Run `python /root/HGETtrack-main/RGBT_workspace/test_HGETtrack.py`

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: 1281718557@qq.com


