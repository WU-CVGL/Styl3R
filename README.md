<h1 align="center"> Styl3R: Instant 3D Stylized Reconstruction for Arbitrary Scenes and Styles </h1>

<div align="center">
  <!-- <a href=https://nickisdope.github.io/Styl3R/ target="_blank"><img src="https://img.shields.io/badge/Project-Page-green.svg" height=22px></a>
  <a href="https://arxiv.org/abs/<ARXIV PAPER ID>" target="_blank"><img src=https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv height=22px></a> -->

<h3 align="center"><a href="https://arxiv.org/abs/2505.21060">Paper</a> | <a href="https://nickisdope.github.io/Styl3R/">Project Page</a> </h3>

📢 The codebase is posted, we are still cleaning it up.

<!-- <img src="assets/teaser_crop-9.gif" width="600" height="258"/> -->

![teaser](assets/teaser_crop-9.gif)

</div>

## 📝 Summary
- We introduce a feed-forward network for 3D stylization that operates on sparse, unposed content images and an arbitrary style image, does not require test-time optimization, and generalizes well to out-of-domain inputs.
- We design a dual-branch network architecture that decouples appearance and structure modeling, effectively enhancing the joint learning of novel view synthesis and 3D stylization.
- Our method achieves state-of-the-art zero-shot 3D stylization performance, surpassing existing zero-shot methods and approximate the efficacy of style-specific optimization techniques.

## 🚧 TODO
- [ ] Release inference code and pretrained models
- [ ] Release gradio demo 
- [ ] Release training code

## 📖 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{wang2025styl3r,
  title={Styl3R: Instant 3D Stylized Reconstruction for Arbitrary Scenes and Styles},
  author={Wang, Peng and Liu, Xiang and Liu, Peidong},
  journal={arXiv preprint arXiv:2505.21060},
  year={2025}
}
