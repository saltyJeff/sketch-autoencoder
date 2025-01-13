# Disentangling EmbEdded Z-Latents

## Abstract
In this paper, I describe a method of disentangling Variational Autoencoder (VAE) latent channels into semantic channels and style channels using Constrastive Language-Imaging Pre-training (CLIP) text and image embeddings.

## Introduction

A VAE encodes images into latent vectors, and decodes latent vectors into images.
An important property of the inputs/outputs of a VAE is that compared to the input images, the latent vectors have more channels, but less spatial information. For example, the Stable Diffusion XL (SDXL) VAE reduces the height/width of an image by a factor of 8, while adding an extra channel (i.e. a $3\times600\times400$ image has a latent representation of $4\times75\times50$, wherein the dimensions correspond to $channels\times height\times width$)

I hypothesize that each channel in the latent vector contains both semantic and style information.

## Method

## Evaluation

### Original Images
<table>
    <tr>
        <td>
            <img src="./img/left.jpg" width=300 />
            <p>Original (Left image)
                <a href="https://negativespace.co/city-street-urban/">CC0 from Bango Architecture & Design</a>
            </p>
        </td>
        <td>
            <img src="./img/right.jpg" width=300 />
            <p>Target (Right image)
                <a href="https://commons.wikimedia.org/wiki/File:Moscow_city_art.jpg"> CC from Viktoria Borodinova</a>
            </p>
        </td>
    </tr>
</table>

### Qualitative Results of Lerping Channels
Each lerp corresponds to replacing the specified channels with a mix of 90% target image and 10% original image.

<table>
    <tr>
        <td>
            <img src="./img/lerp-z-01.jpg" width=300 />
            <p>Semantic lerp</p>
        </td>
        <td>
            <img src="./img/lerp-z-23.jpg" width=300 />
            <p>Style lerp</p>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./img/lerp-03.jpg" width=300 />
            <p>VAE 0-3 lerp</p>
        </td>
        <td>
            <img src="./img/lerp-12.jpg" width=300 />
            <p>VAE 1-2 lerp</p>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./img/lerp-01.jpg" width=300 />
            <p>VAE 0-1 lerp</p>
        </td>
        <td>
            <img src="./img/lerp-23.jpg" width=300 />
            <p>VAE 2-3 lerp</p>
        </td>
    </tr>
    <tr>
        <td>
            <img src="./img/lerp-02.jpg" width=300 />
            <p>VAE 0-2 lerp</p>
        </td>
        <td>
            <img src="./img/lerp-13.jpg" width=300 />
            <p>VAE 1-3 lerp</p>
        </td>
    </tr>
</table>

