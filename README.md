# Lossless coding

## Mothod
This code is the re-implementaion of paper '[LEARNED LOSSLESS IMAGE COMPRESSION WITH A HYPERPRIOR AND DISCRETIZED
GAUSSIAN MIXTURE LIKELIHOODS](https://arxiv.org/abs/2002.01657)'.

The lossless model is based on the lossy model (cheng2020atten) in compressai.

In this method, the output of g_s is changed from x_hat to mean and variance, and the mean variance is used to model x. After that the mean variance is transmitted to the decoding end to complete lossless compression.

## Prepare your dataset
### Train dataset
Here I use 5w images comes from ImageNet to train, the architecture of dataset is:
 
```
root dataset
|__train
|__test
```

### Test dataset
[CLIC val set](http://challenge.compression.cc/tasks/)

[Kodak](http://r0k.us/graphics/kodak/)

## Training 
```python train.py --cuda --save --dataset "your dataset's root path"```

## Eval
I evaluate the performance on kodak/CLIC professional val/CLIC mobile val dataset. 
```python eval.py --path "the path of your dataset"```

## Results
|      | CLICP  | CLICM  | Kodak  |
|------|--------|--------|--------|
| GSM  | 3.582  | 3.465  | 3.840  |
| GMM  | 3.459  | 3.331  | 3.764  |