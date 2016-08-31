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
![fake number](sample_images/gan/individualImage-11.png)
![fake number](sample_images/gan/individualImage-12.png)
![fake number](sample_images/gan/individualImage-13.png)
![fake number](sample_images/gan/individualImage-14.png)
![fake number](sample_images/gan/individualImage-15.png)
![fake number](sample_images/gan/individualImage-16.png)
![fake number](sample_images/gan/individualImage-17.png)
![fake number](sample_images/gan/individualImage-18.png)
![fake number](sample_images/gan/individualImage-19.png)

### InfoGAN

With category 0 active (squiggly snake pretzel):

![fake number](sample_images/infogan/class_0/individualImage.png)
![fake number](sample_images/infogan/class_0/individualImage-1.png)
![fake number](sample_images/infogan/class_0/individualImage-2.png)

With category 1 active (V or 4):

![fake number](sample_images/infogan/class_1/individualImage.png)
![fake number](sample_images/infogan/class_1/individualImage-1.png)
![fake number](sample_images/infogan/class_1/individualImage-2.png)

With category 2 active (3 or 1):

![fake number](sample_images/infogan/class_2/individualImage.png)
![fake number](sample_images/infogan/class_2/individualImage-1.png)
![fake number](sample_images/infogan/class_2/individualImage-2.png)

With category 3 active (unknown or 7s):

![fake number](sample_images/infogan/class_3/individualImage.png)
![fake number](sample_images/infogan/class_3/individualImage-1.png)
![fake number](sample_images/infogan/class_3/individualImage-2.png)

With category 4 active (Candy cane):

![fake number](sample_images/infogan/class_4/individualImage.png)
![fake number](sample_images/infogan/class_4/individualImage-1.png)
![fake number](sample_images/infogan/class_4/individualImage-2.png)

With category 5 active (Cantilevered snake):

![fake number](sample_images/infogan/class_5/individualImage.png)
![fake number](sample_images/infogan/class_5/individualImage-1.png)
![fake number](sample_images/infogan/class_5/individualImage-2.png)

With category 6 active (7 with bubbles):

![fake number](sample_images/infogan/class_6/individualImage.png)
![fake number](sample_images/infogan/class_6/individualImage-1.png)
![fake number](sample_images/infogan/class_6/individualImage-2.png)

With category 7 active (hopping H):

![fake number](sample_images/infogan/class_7/individualImage.png)
![fake number](sample_images/infogan/class_7/individualImage-1.png)
![fake number](sample_images/infogan/class_7/individualImage-2.png)

With category 8 active (lasso 7):

![fake number](sample_images/infogan/class_8/individualImage.png)
![fake number](sample_images/infogan/class_8/individualImage-1.png)
![fake number](sample_images/infogan/class_8/individualImage-2.png)

With category 9 active (C / Tetris piece):

![fake number](sample_images/infogan/class_9/individualImage.png)
![fake number](sample_images/infogan/class_9/individualImage-1.png)
![fake number](sample_images/infogan/class_9/individualImage-2.png)


