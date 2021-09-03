# OutputData
Here there are the results of each code file. 
## Results_similarity_test.xlsx (Results of Similarity_test.py)
This file shows the difference between 03.jpg authentic image and the same image processed by 4 different blur filters. For each filter there are 3 different metrics (SSIM, MSE, RMSE) that mesure the similarity to the real image.
So there are 4 columns:
1. Filter name (Average, Gaussian, Median or Bilateral)
2. SSIM (1st similarity metric)
3. MSE (2nd similarity metric)
4. RMSE (3d similarity metric)
## Results_noise_test.xlsx (Results of Noise_test.py)
This file shows again the similarity between the real and the blurred image. However this is tested for each one of the three images and before the blurring filter there is noise added to each image. Also there are 3 different kinds of noise that are added to the images (Salt and Pepper, Poisson, Gaussian noise). 
So, there are 5 columns:
1. Image id (the processed image)
2. Noise type (Salt and Pepper, Poisson or Gaussian noise)
3. Filter name (Average, Gaussian, Median or Bilateral)
4. SSIM (1st similarity metric)
5. MSE (2nd similarity metric)
