# Diffiner

## Citation
- Valentini-Botinhao, Cassia. (2017). Noisy speech database for training speech enhancement algorithms and TTS models, 2016 [sound]. University of Edinburgh. School of Informatics. Centre for Speech Technology Research (CSTR). https://doi.org/10.7488/ds/2117.

- Diffiner: A Versatile Diffusion-based Generative Refiner for Speech Enhancement.Ryosuke Sawata, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Takashi Shibuya, Shusuke Takahashi, Yuki Mitsufuji.
https://doi.org/10.48550/arXiv.2210.17287

- SEGAN: Speech Enhancement Generative Adversarial Network.Santiago Pascual, Antonio Bonafonte, Joan Serrà.
https://doi.org/10.48550/arXiv.1703.09452

- U-NET : Convolutional Networks for Biomedical Image Segmentation.Olaf Ronneberger, Philipp Fischer, Thomas Brox. https://doi.org/10.48550/arXiv.1505.04597

- NISQA MODEL: G. Mittag, B. Naderi, A. Chehadi, and S. Möller “NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets,” in Proc. Interspeech 2021, 2021.

 ## DESCRIPTION OF THE PROJECT
  
✈The implementation propose a slightly variant of the proposed original architecture, in wich there will be a classical LSGAN approach for the loss in the speech enhancement module and some alternatives in the architectural choiches for the diffusion model.

**Dataset**

The dataset are for clean and noisy tasks, in both case it corresponds to 28 speakers talking for about 5-6 seconds in different noisy eniviroments (for the noisy case) or in a completely silence room(for the clean case).The test set is instead composed by only 2 speakers.

The choiche here is to extract 1/4 of the dataset for the limitation of the RAM in colab enviroment,then preprocess every audio dividing them in segment of 16384 samples with a resampling at 16 kHz for a total of 2892 segment for the training and 206 for the test. For the diffusion model there is another step of the preprocessing consisting in STFT tranformation that has as result a 2 channel tensor (one for the real part and the other one for the imaginary part) with the size [2,256,256].

**Overview of the original project**

First the training of a diffusion-based generative model on
clean speech data. After obtaining results from the SEGAN model, the variance of the noise included in noisy
input at each time-frequency bin is estimated. With the estimate, the proposed refiner generates clean speech on the basis of
the DDRM framework, which utilizes the trained diffusion-based model.

![Screenshot 2024-08-20 004501](https://github.com/user-attachments/assets/417fde5e-24cc-4806-883a-28995ba59391)

**⏰Variant for the loss in the SEGAN model**

The loss will be without the conditioned extra information in order to prove the **classical LSGAN** approach.

**⏰Variant for experiments in the DIFFUSION model**

1.   Classical standard convolution operations substituted by the **depthwise separable convolution**.
   
2. Single head substituted by **n-heads=2** in the attention block.

3. Training with 3 different activation function: **SilU,ReLU and GeLU**.

## SE Module: SEGAN

The **SEGAN** (Speech Enhancement Generative Adversarial Network) architecture is model designed for speech enhancement tasks, particularly to reduce noise in audio signals. It uses a generative adversarial network (GAN) framework, where a generator network aims to produce clean, enhanced speech from noisy input, and a discriminator network attempts to distinguish between the enhanced speech and the original clean speech. Through this **adversarial training**, the SEGAN model effectively learns to improve the quality of noisy speech, making it sound more natural and intelligible.

![Screenshot 2024-08-22 134544](https://github.com/user-attachments/assets/fc9235ed-b440-4f97-b026-d60e60cdfcca)

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

**▶DDPM**

During the **diffusion process** gaussian noise is gradually added in order to achieve the total noise.

Then the Compact U-Net is trained to predict the noise added in order to **reverse the process** and restore the original data.

![Screenshot 2024-09-28 234233](https://github.com/user-attachments/assets/77965cb5-328a-421b-aa27-c0ab9f49c392)

**▶FINAL ALGORITHM FOR THE REFINING TASK**

Here the pseudocode of the final algorithm,for the experiments only the **pattern Diffiner** was considered.

![Screenshot 2024-08-23 150405](https://github.com/user-attachments/assets/5cb6482e-1b57-4820-bdfb-f1b87ca074a5)

**▶RESULTS FOR THE SEGAN MODEL**

For the **SEGAN** model we trained for 65 epochs with the Adam optimizer,a learning rate of 0.001 and a "Reduce on Plateau" scheduler.

After an unstable training we finally reached a smooth behavior in the last 9 epochs and a finally average loss on the test set of 0.0535.

![Screenshot 2024-10-23 010352](https://github.com/user-attachments/assets/7475be4d-f46f-4f94-86de-ddd94f53713e)

**▶RESULTS FOR THE DIFFUSION MODEL**


For the **Diffusion model** we trained with timesteps=50 for 100 epochs with the Adam optimizer,a learning rate of 0.001 and a "Reduce on Plateau" scheduler.

 We trained 3 times in order to experiment with 3 different activation functions : SILU, RELU and GELU

 -  Training with **SILU** 
   
    ![Screenshot 2024-10-23 003202](https://github.com/user-attachments/assets/a24fe2f9-48bb-47cb-83dc-279277e8c2c2)


-  Training with **RELU** 
  
   ![Screenshot 2024-10-23 004406](https://github.com/user-attachments/assets/0a2688b1-4690-4019-ae78-073d4b341570)


-  Training with **GELU** 

   ![Screenshot 2024-10-23 004934](https://github.com/user-attachments/assets/ef6498c7-44d9-4323-8016-52b25b6512ef)

   
In the 3 cases we obtained similar results but with some better performances in the RELU case, in wich the loss seems to converge around 0.1.

Here the result of the final average loss during the test:

1. SILU : 0.1201
  
2. **RELU : 0.0963**

3. GELU : 0.1136

**▶METRICS**

We choose the NISQA metrics in order to evaluaate the performance of the algorithm with these metrics:

- **Overall Quality (MOS_pred)**: Higher values mean better overall speech quality (1-5 scale).

- Noisiness (Noi_pred): Higher values mean more background noise.

- Coloration (Col_pred): Higher values indicate more tonal changes or unnatural sound.

- Discontinuity (Dis_pred): Higher values mean more distortion or breaks in the audio.

- Loudness (Loud_pred): Higher values mean louder speech.


**▶FINAL RESULTS OF THE REFINER**


- Final results with **SILU**
  
  ![silu](https://github.com/user-attachments/assets/1876b114-a61a-4c60-85a7-5230b1758b91)

- Final results with **RELU**

![relu](https://github.com/user-attachments/assets/78d68bc7-2e12-4d82-acd2-78e2d89fa004)

 - Final results with **GELU**

  ![gelu](https://github.com/user-attachments/assets/51298f92-4715-40bc-9b43-f62a0a9f7f74)

  
**▶FINAL CONSIDERATIONS ON THE RESULTS**

In all the experminets the low values of coloration and discontinuity indicates small presence of unnatural sound and distorsion. 

For comparison with the paper we mainly focys on the overall speech quality (MOS_pred) in wich the paper has reached (with the SEGAN model and the Diffiner pattern) a result of 4.372. This result is due to the training of the diffusion model for which they trained the model on a single NVIDIA A100 GPU (40 GB memory) for 7.5 × 10^5 steps, which took about three days. Obviously our results with only 100 epochs and 50 timesteps in the final algorithm (T=200 in the paper) is inevitably lower. 

But the interesting part is that the **RELU experiments** has respected our previsions.In fact ,as we have already said, this was the experiment that showed the better performance during training and the lower final average test loss and ,as a consequence, we reached the better result in terms of MOS_pred scores.












   











