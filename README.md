# ImgBlurFilterComparison
## Comparison of Average, Gaussian, Median and Bilateral filter
The code of this project is seperated in two files, Similarity_test.py and Noise_test.py.
### Similarity_test.py
In this code the similarity between an image before and after blurring is tested. The results are the following:
1. Image output of each filter:![filters_03](https://user-images.githubusercontent.com/89779679/132008071-cacfd612-4545-47dc-9260-6f106d7cac6c.jpg)
2. Similarity of each output image with the initial image (for metrics marked with green the bigger the value the higher the similarity and for metrics marked with red the opposite) :![filters_metrics_03](https://user-images.githubusercontent.com/89779679/132008281-9ae79298-6409-43ed-8ee7-a2b9f873d772.jpg)

As can be seen from the values, all metrics conclude that the 4 filters cause a distortion in the image, with the largest deterioration being caused by the Averaging filter and the rest following in descending order: Median filter, Gaussian filter, Bilateral filter. Therefore, Bilateral causes the smallest alteration, maintaining more than 99% similarity with the original image, according to the SSIM metric.
### Noise_test.py
In this code the perfomance of each blur filter in noise reduction is tested. This is done for all 3 images with 3 different types of noise (Salt and Pepper, Gaussian, Poisson). The results are the following:
1. SSIM metric results for each image, each type of noise and each blurring filter:![ssim](https://user-images.githubusercontent.com/89779679/132010723-2d4203cb-d15d-401a-a0c5-4082ccf9e6ad.jpg)
2. MSE metric results for each image, each type of noise and each blurring filter:![mse](https://user-images.githubusercontent.com/89779679/132010898-eb145b17-b69b-462f-bca5-3a7d327c6c77.jpg)

According to both metrics Bilateral filter has the best performance in case of low noise, such as poisson (poisson noise was added in a smaller percentage than the other types of noise). However, it has the worst results when the noise is higher, such as average and salt and pepper noise.This seems logical, since this filter does not distort the image as much as the rest, because when it modifies the value of a pixel, it takes into account the values of neighboring pixels with weight corresponding not only with their proximity in terms of space but also in terms of color intensity. Therefore when the noise does not cause very strong changes it is much better manageable by this filter than by any other. The other filters that do not have this criterion of color intensity, although they eliminate the noise to a certain extent, at the same time they distort more the real elements of the image. 
In case of Salt and Pepper noise Median filter has the best performance. The fact that this filter is based on the intermediate contributes to that, because it is not easily affected by extreme values, such as those added by the Salt and Pepper noise.
