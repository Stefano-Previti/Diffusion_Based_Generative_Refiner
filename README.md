# Diffiner

## Citation
- Valentini-Botinhao, Cassia. (2017). Noisy speech database for training speech enhancement algorithms and TTS models, 2016 [sound]. University of Edinburgh. School of Informatics. Centre for Speech Technology Research (CSTR). https://doi.org/10.7488/ds/2117.

- Diffiner: A Versatile Diffusion-based Generative Refiner for Speech Enhancement.Ryosuke Sawata, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Takashi Shibuya, Shusuke Takahashi, Yuki Mitsufuji.
https://doi.org/10.48550/arXiv.2210.17287

- SEGAN: Speech Enhancement Generative Adversarial Network.Santiago Pascual, Antonio Bonafonte, Joan Serrà.
https://doi.org/10.48550/arXiv.1703.09452

- U-NET : Convolutional Networks for Biomedical Image Segmentation.Olaf Ronneberger, Philipp Fischer, Thomas Brox. https://doi.org/10.48550/arXiv.1505.04597

 ## Overview of the project
  
First the training of a diffusion-based generative model on clean speech data. After obtaining results from an arbitrary preceding SE module, the variance of the noise included in noisy input at each time-frequency bin is estimated. With the estimate, the proposed refiner generates clean speech on the basis of the DDRM framework, which utilizes the pre-trained diffusion-based model.

![Screenshot 2024-08-20 004501](https://github.com/user-attachments/assets/417fde5e-24cc-4806-883a-28995ba59391)

## SE Module: SEGAN

The **SEGAN** (Speech Enhancement Generative Adversarial Network) architecture is model designed for speech enhancement tasks, particularly to reduce noise in audio signals. It uses a generative adversarial network (GAN) framework, where a generator network aims to produce clean, enhanced speech from noisy input, and a discriminator network attempts to distinguish between the enhanced speech and the original clean speech. Through this **adversarial training**, the SEGAN model effectively learns to improve the quality of noisy speech, making it sound more natural and intelligible.
![Screenshot 2024-08-22 134544](https://github.com/user-attachments/assets/fc9235ed-b440-4f97-b026-d60e60cdfcca)

**▶SCALING OF THE ARCHITECTURE**

Here there is a **scaled** model in order to achieve a better performance in terms of time-consuming during training.

The model goes up to a maximum of 256 channels instead of the 1024 reached by the paper.

**⏰VARIANT FOR THE LOSS**

The loss will be without the conditioned extra information in order to prove the **classical LSGAN** approach.

Here showed the (general) equation of the loss using the **least-squares GAN** (LSGAN) approach obtained withthe least-squares function.

![Screenshot 2024-10-15 224526](https://github.com/user-attachments/assets/a9d86b2c-09eb-4bbf-b2ff-f0a2b0c4e84e)

In this  specific case the generic parameters (a,b,c) are substituted with binary coding(1 for real, 0 for
fake).

Moreover there is an **L1 regularization** added in the **generator loss** as suggested by the paper

**Adaptation for the speech enhancement task**

1.   The concatenation of the noisy with the clean signal should be identified from the discriminator as True.

2.   The concatenation of the noisy with the enhanced signal should be identified from the discriminator as False.

3. The generator want to make True the identification of the discriminator about the concatenation of the noisy signal with the enhanced signal.

![Screenshot 2024-10-07 004720](https://github.com/user-attachments/assets/3f9254e6-8d39-44d7-80d9-a1ec63a4ef03)

## **DIFFUSION BASED GENERATIVE MODEL**

The autoencoder U-NET, here a sketch of the original model.

![Screenshot 2024-08-23 165637](https://github.com/user-attachments/assets/2c9b2703-64bd-4550-8b19-2e8dca8dcaa5)

**▶ U-NET MODEL FOR DENOISING TASK: MAIN DIFFERENCES**

1.   **Residual block**: instead of simple sequences of convolutional operations with activation functions,the residual block shows the normalization, the residual connection and the time embedding added after the first convolution.

2.   **Attention block**: the block performs an attention and a skip connection in order to capture more information than the standard model.

**⏰VARIANT FOR THE 2D CONVOLUTIONAL OPERATIONS**

![Screenshot 2024-09-28 131049](https://github.com/user-attachments/assets/ca189c97-a92b-4614-9862-a93e54195bc6)

The standard convolution operation combines the values of all the input channels.

The total number of parameters in a standard convolutional layer can be expressed as:

**Total Parameters=(Wk ⋅ Hk ⋅ din ⋅ dout) + dout**

**Depthwise Separable Convolutions**

The depthwise separable convolution approach is a different, more efficient operation.

It is a two-step operation: first, a depthwise convolution, followed by a 1 x 1 pointwise convolution.

![Screenshot 2024-09-28 131056](https://github.com/user-attachments/assets/a745590f-3efc-47ba-a819-67e7fd783e7f)

The depthwise convolution does not combine the input channels. It convolves on each channel separately so **each channel** gets its own set of weights.

![Screenshot 2024-09-28 131102](https://github.com/user-attachments/assets/3b84f1c4-6f81-4275-b5ee-71109741ab6c)

The pointwise convolution is essentially the same as a standard convolution, except using a 1 x 1 kernel. This operation just adds up the channels from the depthwise convolution as a weighted sum.

**Parameters Calculation**

1)For ***depthwise convolution***, the number of parameters is calculated as follows:

Parameters(Depthwise)=Wk⋅Hk⋅din

2)For  ***pointwise convolution***, which combines the outputs from the depthwise convolution:

Parameters(Pointwise)=din⋅dout

**Total Parameters=(Wk ⋅ Hk ⋅ din)+ (din ⋅ dout)**

**⏰FINAL MODEL VARIANTS FOR EXPERIMENTS**

1.   Classical 2D standard convolution operations substituted by the **depthwise separable convolution**.
  
2. Single head substituted by **n-heads=2** in the attention block.

3. Prove 3 different activation function: **SilU,ReLU and GeLU**.

**▶DDPM**

During the **diffusion process** gaussian noise is gradually added in order to achieve the total noise.

Then the Compact U-Net is trained to predict the noise added in order to **reverse the process** and restore the original data.

![Screenshot 2024-09-28 234233](https://github.com/user-attachments/assets/77965cb5-328a-421b-aa27-c0ab9f49c392)

**▶FINAL ALGORITHM FOR THE REFINING TASK**

![Screenshot 2024-08-23 150405](https://github.com/user-attachments/assets/5cb6482e-1b57-4820-bdfb-f1b87ca074a5)









