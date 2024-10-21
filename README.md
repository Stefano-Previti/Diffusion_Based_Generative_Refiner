# Diffiner

## Citation
- Valentini-Botinhao, Cassia. (2017). Noisy speech database for training speech enhancement algorithms and TTS models, 2016 [sound]. University of Edinburgh. School of Informatics. Centre for Speech Technology Research (CSTR). https://doi.org/10.7488/ds/2117.

- Diffiner: A Versatile Diffusion-based Generative Refiner for Speech Enhancement.Ryosuke Sawata, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Takashi Shibuya, Shusuke Takahashi, Yuki Mitsufuji.
https://doi.org/10.48550/arXiv.2210.17287

- SEGAN: Speech Enhancement Generative Adversarial Network.Santiago Pascual, Antonio Bonafonte, Joan Serrà.
https://doi.org/10.48550/arXiv.1703.09452

- U-NET : Convolutional Networks for Biomedical Image Segmentation.Olaf Ronneberger, Philipp Fischer, Thomas Brox. https://doi.org/10.48550/arXiv.1505.04597

- ## Overview of the project
  
First the training of a diffusion-based generative model on clean speech data. After obtaining results from an arbitrary preceding SE module, the variance of the noise included in noisy input at each time-frequency bin is estimated. With the estimate, the proposed refiner generates clean speech on the basis of the DDRM framework, which utilizes the pre-trained diffusion-based model.

![Screenshot 2024-08-20 004501](https://github.com/user-attachments/assets/417fde5e-24cc-4806-883a-28995ba59391)

