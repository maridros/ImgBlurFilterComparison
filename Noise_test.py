# Drosou Maria
# Department of Informatics And Computer Engineering, University of West Attica
# e-mail: cs151046@uniwa.gr
# A.M.: 151046

import skimage.metrics
from skimage.io import imread_collection
from skimage.util import random_noise
import cv2
import numpy as np
import time
import pandas as pd  # for excel file


# define function to calculate the similarity between two images using two metrics
def compare_imgs(grey_img1, grey_img2, run_time=0.0, print_flag=0, filter_name=""):
    # Metric 1: compute the Structural Similarity Index (SSIM) between the two images
    (ssim, _) = skimage.metrics.structural_similarity(grey_img1, grey_img2, full=True)
    # Metric 2: compute the Mean Squared Error (MSE) between the two images
    mse = skimage.metrics.mean_squared_error(grey_img1, grey_img2)
    if print_flag:
        # Print the results
        print(" ..", filter_name, "time was: {:.8f}".format(run_time), " seconds. ")
        print(" ... SSIM score: {:.4f}".format(ssim))
        print(" ... MSE score: {:.4f}".format(mse))
    return ssim, mse


# define function to store results of each image in a dataframe
def store_data(df, image_id, noise_type, a_ssim, a_mse, g_ssim, g_mse, m_ssim, m_mse, b_ssim, b_mse):
    df = df.append({'Image ID': image_id, 'Noise type': noise_type,
                    'Filter Name': "Average Filter", 'SSIM value': g_ssim,
                    'MSE value': g_mse}, ignore_index=True)
    df = df.append({'Image ID': image_id, 'Noise type': noise_type,
                    'Filter Name': "Gaussian Blur Filter", 'SSIM value': a_ssim,
                    'MSE value': a_mse}, ignore_index=True)
    df = df.append({'Image ID': image_id, 'Noise type': noise_type,
                    'Filter Name': "Median Filter", 'SSIM value': m_ssim,
                    'MSE value': m_mse}, ignore_index=True)
    df = df.append({'Image ID': image_id, 'Noise type': noise_type,
                    'Filter Name': "Bilateral Filter", 'SSIM value': b_ssim,
                    'MSE value': b_mse}, ignore_index=True)
    return df


# define function to apply several smoothing filters
def apply_filters(img_BGR, img_GRAY, height, width):
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

    a_ssim, a_mse, = compare_imgs(img_GRAY, blur_GRAY, tmpRunTime,
                                  print_flag=1, filter_name="Averaging blur filter")

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

    g_ssim, g_mse = compare_imgs(img_GRAY, blur_GRAY, tmpRunTime,
                                 print_flag=1, filter_name="Gaussian blur filter")

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

    m_ssim, m_mse = compare_imgs(img_GRAY, blur_GRAY, tmpRunTime,
                                 print_flag=1, filter_name="Median filter")

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

    b_ssim, b_mse = compare_imgs(img_GRAY, blur_GRAY, tmpRunTime,
                                 print_flag=1, filter_name="Bilateral blur filter")

    cv2.waitKey()
    cv2.destroyWindow("Averaging Filter")
    cv2.destroyWindow("Gauss blur Filter")
    cv2.destroyWindow("Median Filter")
    cv2.destroyWindow("Bilateral Filter")

    return a_ssim, a_mse, g_ssim, g_mse, m_ssim, m_mse, b_ssim, b_mse


# define function to add noise to image and attempt to fix it with several filters
def add_noise_and_test_filters(image, img_id, data):
    img_BGR = image
    img_GRAY = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)

    # get image dimensions
    width, height = img_GRAY.shape

    # and illustrate the results
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", img_BGR)
    cv2.resizeWindow("Original Image", height, width)

    cv2.waitKey()

    # add noise to image

    # a. gaussian

    print("Applying gaussian noise")

    img_gauss_noise_BGR = random_noise(img_BGR, mode='gaussian', clip=True, var=0.2)

    # The above function returns a floating-point image on the range [0, 1],
    # thus we changed it to 'uint8' and from [0,255]
    img_gauss_noise_BGR = np.array(255 * img_gauss_noise_BGR, dtype='uint8')

    img_gauss_noise_GRAY = cv2.cvtColor(img_gauss_noise_BGR, cv2.COLOR_BGR2GRAY)

    # illustrate the results
    cv2.namedWindow("Image with gaussian noise", cv2.WINDOW_NORMAL)
    cv2.imshow("Image with gaussian noise", img_gauss_noise_BGR)
    cv2.resizeWindow("Image with gaussian noise", height, width)

    print("Demonstrating the noise reduction capabilities for each of the filters")

    ssim_score, mse_score = compare_imgs(img_GRAY, img_gauss_noise_GRAY)

    print(" .. Currently:")
    print(" ... SSIM score between original and noise image is: {:.4f}".format(ssim_score))
    print(" ... MSE score between original and noise image is: {:.4f}".format(mse_score))

    print(" .. Attempting to fix results, using different filters")

    cv2.waitKey()

    print('Illustrating the effects of various filters, in terms of time and information loss')

    a_ssim_score, a_mse_score, g_ssim_score, g_mse_score, \
    m_ssim_score, m_mse_score, b_ssim_score, b_mse_score \
        = apply_filters(img_gauss_noise_BGR, img_GRAY, height, width)

    data = store_data(data, img_id, "Gaussian noise", a_ssim_score,
                      a_mse_score, g_ssim_score, g_mse_score,
                      m_ssim_score, m_mse_score, b_ssim_score, b_mse_score)

    # b. salt and pepper noise

    print("Applying salt and pepper noise")

    img_sp_noise_BGR = random_noise(img_BGR, mode='s&p', amount=0.3)

    # The above function returns a floating-point image on the range [0, 1],
    # thus we changed it to 'uint8' and from [0,255]
    img_sp_noise_BGR = np.array(255 * img_sp_noise_BGR, dtype='uint8')

    img_sp_noise_GRAY = cv2.cvtColor(img_sp_noise_BGR, cv2.COLOR_BGR2GRAY)

    # illustrate the results
    cv2.namedWindow("Image with salt and pepper noise", cv2.WINDOW_NORMAL)
    cv2.imshow("Image with salt and pepper noise", img_sp_noise_BGR)
    cv2.resizeWindow("Image with salt and pepper noise", height, width)

    print("Demonstrating the noise reduction capabilities for each of the filters")

    ssim_score, mse_score = compare_imgs(img_GRAY, img_sp_noise_GRAY)

    print(" .. Currently:")
    print(" ... SSIM score between original and noise image is: {:.4f}".format(ssim_score))
    print(" ... MSE score between original and noise image is: {:.4f}".format(mse_score))

    print(" .. Attempting to fix results, using different filters")

    cv2.waitKey()

    print('Illustrating the effects of various filters, in terms of time and information loss')

    a_ssim_score, a_mse_score, g_ssim_score, g_mse_score, \
    m_ssim_score, m_mse_score, b_ssim_score, b_mse_score = \
        apply_filters(img_sp_noise_BGR, img_GRAY, height, width)

    data = store_data(data, img_id, "Salt and pepper noise", a_ssim_score,
                      a_mse_score, g_ssim_score, g_mse_score,
                      m_ssim_score, m_mse_score, b_ssim_score, b_mse_score)

    # c. poisson noise

    print("Applying poisson noise")

    img_poisson_noise_BGR = random_noise(img_BGR, mode='poisson', clip=True)

    # The above function returns a floating-point image on the range [0, 1],
    # thus we changed it to 'uint8' and from [0,255]
    img_poisson_noise_BGR = np.array(255 * img_poisson_noise_BGR, dtype='uint8')

    img_poisson_noise_GRAY = cv2.cvtColor(img_poisson_noise_BGR, cv2.COLOR_BGR2GRAY)

    # illustrate the results
    cv2.namedWindow("Image with poisson noise", cv2.WINDOW_NORMAL)
    cv2.imshow("Image with poisson noise", img_poisson_noise_BGR)
    cv2.resizeWindow("Image with poisson noise", height, width)

    print("Demonstrating the noise reduction capabilities for each of the filters")

    ssim_score, mse_score = compare_imgs(img_GRAY, img_poisson_noise_GRAY)

    print(" .. Currently:")
    print(" ... SSIM score between original and noise image is: {:.4f}".format(ssim_score))
    print(" ... MSE score between original and noise image is: {:.4f}".format(mse_score))

    print(" .. Attempting to fix results, using different filters")

    cv2.waitKey()

    print('Illustrating the effects of various filters, in terms of time and information loss')

    a_ssim_score, a_mse_score, g_ssim_score, g_mse_score, \
    m_ssim_score, m_mse_score, b_ssim_score, b_mse_score = \
        apply_filters(img_poisson_noise_BGR, img_GRAY, height, width)

    data = store_data(data, img_id, "Poisson noise", a_ssim_score,
                      a_mse_score, g_ssim_score, g_mse_score,
                      m_ssim_score, m_mse_score, b_ssim_score, b_mse_score)

    cv2.destroyAllWindows()

    return data


# Create a dataframe for the results
resultsDf = pd.DataFrame(columns=['Image ID', 'Noise type', 'Filter Name',
                                  'SSIM value', 'MSE value'])

# getting path of the images
col_dir = 'assets/*.jpg'

# creating a collection with all the images
col_RGB = imread_collection(col_dir)
num_of_imgs = len(col_RGB)

# For each image run function add_noise_and_test_filters and get results to the dataframe
i = 1
for img in col_RGB:
    print(64 * "-")
    print("Image", str(i) + "/" + str(num_of_imgs))
    print(64 * "-")
    img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    resultsDf = add_noise_and_test_filters(img_BGR, i, resultsDf)
    i = i + 1

# export the results of all filters to excel file
writer = pd.ExcelWriter('OutputData/Results_noise_test.xlsx')
resultsDf.to_excel(writer, 'Q2', index=False)
writer.save()
writer.close()
