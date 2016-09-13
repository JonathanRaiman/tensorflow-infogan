InfoGAN
-------

This repository contains a straightforward implementation of [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) trained to fool a discriminator that sees real MNIST images, along with [Mutual Information Generative Adversarial Networks (InfoGAN)](https://arxiv.org/abs/1606.03657).

## Usage

* Install tensorflow

Then run for GAN:

```
python3 infogan/__init__.py
```

And InfoGAN:

```
python3 infogan/__init__.py --infogan
```

## Visualization

To see samples from the model during training you can use Tensorboard as follows:

```
tensorboard --logdir MNIST_v1_log/
```

## Expected Result

### GAN

You should now see images like these show up:

![fake number](sample_images/gan/individualImage.png)
![fake number](sample_images/gan/individualImage-1.png)
![fake number](sample_images/gan/individualImage-2.png)
![fake number](sample_images/gan/individualImage-3.png)
![fake number](sample_images/gan/individualImage-4.png)
![fake number](sample_images/gan/individualImage-5.png)
![fake number](sample_images/gan/individualImage-6.png)
![fake number](sample_images/gan/individualImage-7.png)
![fake number](sample_images/gan/individualImage-8.png)
![fake number](sample_images/gan/individualImage-9.png)
![fake number](sample_images/gan/individualImage-10.png)

### InfoGAN

On tensorboard you should see the following properties emerge:

![variations](sample_images/infogan/variations.png)

