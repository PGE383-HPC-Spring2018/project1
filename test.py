#!/usr/bin/env python

from project1 import Lorenz
import numpy as np

import skimage
import skimage.measure
import skimage.transform
import cv2


def test_Lorenz_1():
    
    sol = Lorenz(10, 8/3., 14)
    ans1 = sol.solve(0)[:3]
    ans2 = sol.solve(0)[-1]
    np.testing.assert_allclose(ans1, np.array([[0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
       [9.49105159e-02, 9.96784895e-01, 4.76920797e-04],
       [1.81103514e-01, 1.00618983e+00, 1.83590082e-03]]), atol=0.001) 
    np.testing.assert_allclose(ans2, np.array([-5.88784058, -5.88784058, 13.]), atol=0.001) 

    
def test_Lorenz_2():
    
    sol = Lorenz(10, 8/3., 28)
    ans1 = sol.solve(0)[:4]
    ans2 = sol.solve(0)[-1]
    np.testing.assert_allclose(ans1, np.array([[0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
       [9.51319372e-02, 1.00353668e+00, 4.79098355e-04],
       [1.82791662e-01, 1.03242460e+00, 1.86879673e-03],
       [2.66004060e-01, 1.08475943e+00, 4.16815796e-03]]), atol=0.001) 
    np.testing.assert_allclose(ans2, np.array([ 6.38177035, 10.07082235, 16.93460356]), atol=0.001) 

def test_plot():
    
    p = Lorenz(0,0,0)
    p.plot('lorenz.png')
    
    gold_image = cv2.imread('lorenz_gold.png')
    test_image = cv2.imread('lorenz.png')
    
    test_image_resized = skimage.transform.resize(test_image, 
                                                  (gold_image.shape[0], gold_image.shape[1]), 
                                                  mode='constant')
    
    ssim = skimage.measure.compare_ssim(skimage.img_as_float(gold_image), test_image_resized, multichannel=True)
    assert ssim >= 0.8
