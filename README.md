<p align = "center">    
<img  src="paper_logo.png" width="400" />
</p>


# <center> **Read Before Grounding: Scene Knowledge Visual Grounding via Multi-step parsing** </center>

This is the official code implement of COLING 2025 paper **Read Before Grounding: Scene Knowledge Visual Grounding via Multi-step parsing**

## **Data Preparation**

Download data follow https://github.com/zhjohnchan/SK-VG

Unzip the file to the current folder after the data download is complete

## **Main Experiment**
First, you should generate the visual descriptor:

`python qwen_api.py # you may need adjust the data path`

then you could use these visual descriptors evaluate multimodal models.

- We have also prepared visual descriptors for each experiment in the `reading_results/`,

    `ours_{}.json` is the generated result of the main experiment;
  
    `ours_{}_baseline.json` is the generated result of the ablation study;
  
    `ours_{}_glm.json` is the generated result of the analysis.

## **Ablation Study**
`python qwen_api_baseline.py # you may need adjust the data path`

## **Analysis**
`python glm4_flash.py # you may need adjust the data path`

## Acknowledgement
- [UNINEXT](https://github.com/MasterBin-IIAU/UNINEXT)
- [OFA](https://github.com/OFA-Sys/OFA)
- [Shikra](https://github.com/shikras/shikra)
- [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE)
- [InternVL2](https://github.com/OpenGVLab/InternVL)
- [GroundingGPT](https://github.com/lzw-lzw/GroundingGPT)
- [GroundVLP](https://github.com/om-ai-lab/GroundVLP)