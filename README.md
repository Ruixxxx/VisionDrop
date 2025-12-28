## VisionDrop

This repository provides the implementation for our paper: "[_Rethinking Visual Token Reduction in LVLMs under Cross-modal Misalignment_](https://arxiv.org/abs/2506.22283)", __AAAI 2026__.

### 🧩 Method

<p align="center">
  <img src="figures/VisionDrop_poster_01.png" width="900">
</p>

### 📦 Environment Setup

1. Install necessary packages.
```Shell
conda create -n vdrop python=3.10 -y
conda activate vdrop

# Install dependencies
pip install --upgrade pip
pip install -e .
```

2. (Optional) Install FlashAttention for further inference acceleration.
```Shell
pip install flash-attn --no-build-isolation
```

### ⚙️ Key Arguments
Our method performs progressive visual token reduction at both the visual encoder and LLM decoding phase. The main arguments are:

- `--dominant  '42' `: Number of dominant tokens retained from the visual encoder.
- `--contextual  '6' `: Number of contextual tokens retained alongside dominant ones from the visual encoder.
- `--layer_list  '[8,16,24]' `: LLM layers after which token reduction is applied.
- `--image_token_list "[[30,5],[22,4],[16,3]]" `: Token retention schedule per LLM layer, formatted as list of [dominant, contextual].

These example settings correspond to an average token retention of 32 tokens.

### 🚀 Efficient Inference

We follow the original evaluation in [LLaVA](https://github.com/haotian-liu/LLaVA) on 9 image understanding benchmarks.

Before evaluation, prepare the datasets following the LLaVA [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) instructions, 
and download LLaVA-1.5-7B checkpoints from [Hugging Face](https://huggingface.co/liuhaotian/llava-v1.5-7b).

We provide the evaluation scripts for each benchmark:

```Shell
bash scripts/v1_5/visiondrop_eval/${DATASET}.sh
```

### 🔗 Citation

If you find this project useful in your research, please consider citing:

    @article{xu2025visiondrop,
        author    = {Rui Xu and Yunke Wang and Yong Luo and Bo Du},
        title     = {Rethinking Visual Token Reduction in LVLMs under Cross-modal Misalignment},
        journal   = {arXiv preprint arXiv:2506.22283},
        year      = {2025},
    }

### ❤️ Acknowledgments

This work builds upon several excellent open-source projects:
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [PyramidDrop](https://github.com/Cooperx521/PyramidDrop)
- [VisionZip](https://github.com/dvlab-research/VisionZip)

Thanks for the original authors for their contributions to the community.
