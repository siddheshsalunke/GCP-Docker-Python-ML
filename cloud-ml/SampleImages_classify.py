"""

This program uses a supervised classification approach to return
a list of URLs of daytime images, and a list of URL of nighttime images,
stored in a Google Cloud storage bucket. 

Usage:  python findSampleImages_classify.py

"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from PIL import Image
import pickle
import urllib
import urllib.request
import shutil
import os
import sys


def day_night_image_classify_url(txt_url, repo_url, model, feat_scaler):
    """
    This function uses the previously obtained model and transformations
    to perform day/night scene classification of repository images.
    The repository images do not need to be stored in a local directory.

    Args:
        txt_url: The URL of the text file containing the list of repository
                images.
        repo_url: The URL of the repository.
        model: The classification model
        feat_scaler: The feature normalization transformation to be applied to
                each normalized histogram.

    Returns:
        List of strings: A list containing strings of the names of images
                    classified as daytime.

    """

    day_images = list()
    night_images = list()
    cnt = 1
    fid = urllib.request.urlopen(txt_url)
    img_all_bytes = fid.readlines()
    img_all_bytes = img_all_bytes
    for img_byte in img_all_bytes:
        img_file = img_byte.decode('utf-8').strip()
        response = urllib.request.urlopen( repo_url + img_file)
        raw = Image.open(response)
        if len(np.array(raw).shape) == 3:
            # Extract R, G, B components, and obtain grayscale
            # pixel intensity from the components
            red = np.array(raw)[:, :, 0]
            grn = np.array(raw)[:, :, 1]
            blu = np.array(raw)[:, :, 2]
            grayscale = 0.2126 * red + 0.7152 * grn + 0.0722 * blu
        else:
            # Image was grayscale to begin with
            grayscale = np.array(raw)

        im_flat = grayscale.reshape(1, grayscale.shape[0]*grayscale.shape[1])

        # Extract pixels for top third of image
        im_flat_topthird = im_flat[0][0:int(len(im_flat[0])/3)]

        # Generate histogram of the image pixel intensities
        im_hist = np.histogram(im_flat_topthird, bins=20, range=(0, 255))[0]

        feats_test = im_hist/float(np.sum(im_hist))
        feats_test = feat_scaler.transform(feats_test.reshape(1, -1))

        # Generate predicted class of test image
        yhat = model.predict(feats_test)
        if yhat == 1:
            day_images.append(img_file)
            urllib.request.urlretrieve(os.path.join(repo_url, img_file), './classified_images/day/' + img_file)
        else:
            night_images.append(img_file)
            urllib.request.urlretrieve(os.path.join(repo_url, img_file), './classified_images/night/' + img_file)

        print('Classifying image', str(cnt), ':', img_file)

        cnt += 1

    return day_images, night_images


def load_models():
    """

    This function loads the classification model and feature scaling
    transformations from the local disk. This allows the user to run this
    program without having to re-train the model and transformations
    each time (if the same model and transformations are to be used).

    Args:
        None

    Returns:
        model: The classification model
        feat_scaler: The feature-scaling transformation

    """

    # Load model and transformations to disk (if needed)
    model = pickle.load(open('saved_svm_model', 'rb'), encoding='utf-8')
    feat_scaler = pickle.load(open('saved_sc', 'rb'))

    return model, feat_scaler


def main():
    
    repo_url = 'https://storage.googleapis.com/cloud-ml-model45/'
    txt_url = 'https://storage.googleapis.com/cloud-ml-model-txt/list_file.txt'

    shutil.rmtree('./classified_images', ignore_errors=True)
    os.makedirs('./classified_images/day')
    os.makedirs('./classified_images/night')

    # Load model and transformations to disk (if needed)
    model, feat_scaler = load_models()

    # Use trained model to classify images in repository
    day_images, night_images = day_night_image_classify_url(txt_url, repo_url,
                                                  model, feat_scaler)


 
    
    print('--- List of URLs of day-time images ---')
    for day_image in day_images:
        print('https://storage.googleapis.com/cloud-ml-model45/' + day_image)
    print('--- List of URLs of night-time images ---')
    for night_image in night_images:
        print('https://storage.googleapis.com/cloud-ml-model45/' + night_image)


if __name__ == '__main__':
    main()
