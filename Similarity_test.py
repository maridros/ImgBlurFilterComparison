# Drosou Maria
# Department of Informatics And Computer Engineering, University of West Attica
# e-mail: cs151046@uniwa.gr
# A.M.: 151046

# required packages
import cv2
import random
import time
import skimage.metrics
import pandas as pd  # for excel file


# define function to calculate the similarity between two images using three measures
def compare_imgs(grey_img1, grey_img2, filter_name, run_time):
    # Metric 1: compute the Structural Similarity Index (SSIM) between the two images
    (ssim, _) = skimage.metrics.structural_similarity(grey_img1, grey_img2, full=True)
    # Metric 2: compute the Mean Squared Error (MSE) between the two images
    mse = skimage.metrics.mean_squared_error(grey_img1, grey_img2)
    # Metric 3: compute the Normalized Root Mean Squared Error (NRMSE) between the two images
    nrmse = skimage.metrics.normalized_root_mse(grey_img1, grey_img2,
                                                normalization='euclidean')
    # Print the results
    print(" ..", filter_name, "time was: {:.8f}".format(run_time), " seconds. ")
    print(" ... SSIM score: {:.4f}".format(ssim))
    print(" ... MSE score: {:.4f}".format(mse))
    print(" ... NRMSE score: {:.4f}".format(nrmse))
    return ssim, mse, nrmse


def store_data(df, ssim, mse, nrmse, filtername):
    df = df.append({'Filter Name': filtername, 'SSIM value': ssim,
                    'MSE value': mse, 'NRMSE value': nrmse},
                   ignore_index=True)
    return df


# Create a dataframe for the results
resultsDf = pd.DataFrame(columns=['Filter Name', 'SSIM value', 'MSE value', 'NRMSE value'])


# load the image
random.seed(6)
img_num = random.randint(1, 3)
img_name = "assets/0" + str(img_num) + ".jpg"  # choose a random image
# img_name = "assets/03.jpg"  # can be used if we want to load a certain image (03.jpg)
img_BGR = cv2.imread(img_name)
img_GRAY = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)

# get image dimensions
width, height = img_GRAY.shape

print("'" + img_name + "'", "loaded")

# and illustrate the results
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", img_BGR)
cv2.resizeWindow("Original Image", height, width)

cv2.namedWindow("Grayscale Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Grayscale Original Image", img_GRAY)
cv2.resizeWindow("Grayscale Original Image", height, width)

cv2.waitKey()
cv2.destroyWindow("Grayscale Original Image")


# apply some filters and compare change to the original
print('Illustrating the effects of various filters, in terms of time and information loss')

# a. averaging filter
t = time.time()
blur = cv2.blur(img_BGR, (5, 5))
tmpRunTime = time.time() - t

# illustrate the results
cv2.namedWindow("Averaging Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Averaging Filter", blur)
cv2.resizeWindow("Averaging Filter", height, width)

# calculate the similarity between the images
blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

ssim_score, mse_score, nrmse_score = compare_imgs(img_GRAY, blur_GRAY, "Averaging blur filter",
                                                  tmpRunTime)
resultsDf = store_data(resultsDf, ssim_score, mse_score, nrmse_score, "Averaging blur filter")


cv2.waitKey()


# b. gaussian filter
t = time.time()
blur = cv2.GaussianBlur(img_BGR, (5, 5), 0)
tmpRunTime = time.time() - t

# illustrate the results
cv2.namedWindow("Gauss blur Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Gauss blur Filter", blur)
cv2.resizeWindow("Gauss blur Filter", height, width)

# calculate the similarity between the images
blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

ssim_score, mse_score, nrmse_score = compare_imgs(img_GRAY, blur_GRAY, "Gaussian blur filter",
                                                  tmpRunTime)
resultsDf = store_data(resultsDf, ssim_score, mse_score, nrmse_score, "Gaussian blur filter")


cv2.waitKey()

# c. median filter
t = time.time()
blur = cv2.medianBlur(img_BGR, 5)
tmpRunTime = time.time() - t

# illustrate the results
cv2.namedWindow("Median Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Median Filter", blur)
cv2.resizeWindow("Median Filter", height, width)

# calculate the similarity between the images
blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

ssim_score, mse_score, nrmse_score = compare_imgs(img_GRAY, blur_GRAY, "Median blur filter",
                                                  tmpRunTime)
resultsDf = store_data(resultsDf, ssim_score, mse_score, nrmse_score, "Median blur filter")


cv2.waitKey()

# d. bilateral filter
t = time.time()
blur = cv2.bilateralFilter(img_BGR, 9, 5, 5)
tmpRunTime = time.time() - t

# illustrate the results
cv2.namedWindow("Bilateral Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Bilateral Filter", blur)
cv2.resizeWindow("Bilateral Filter", height, width)

# calculate the similarity between the images
blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

ssim_score, mse_score, nrmse_score = compare_imgs(img_GRAY, blur_GRAY, "Bilateral blur filter",
                                                  tmpRunTime)
resultsDf = store_data(resultsDf, ssim_score, mse_score, nrmse_score, "Bilateral blur filter")


cv2.waitKey()

# export the results of all filters to excel file
writer = pd.ExcelWriter('OutputData/Results_similarity_test.xlsx')
resultsDf.to_excel(writer, 'Q1', index=False)
writer.save()
writer.close()

cv2.destroyAllWindows()
