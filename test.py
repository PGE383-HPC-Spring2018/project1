#/usr/bin/env python
#
# Copyright 2020-2021 John T. Foster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import nbconvert
import os

import skimage
import skimage.measure
import skimage.transform
import cv2
import warnings

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


with open("project1.ipynb") as f:
    exporter = nbconvert.PythonExporter()
    python_file, _ = exporter.from_file(f)


with open("project1.py", "w") as f:
    f.write(python_file)


from project1 import Lorenz

class TestSolution(unittest.TestCase):

    def test_Lorenz(self):
        
        sol = Lorenz(10, 8/3., 14)
        ans1 = sol.solve(0)[:3]
        ans2 = sol.solve(0)[-1]
        np.testing.assert_allclose(ans1, np.array([[0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
           [9.49105159e-02, 9.96784895e-01, 4.76920797e-04],
           [1.81103514e-01, 1.00618983e+00, 1.83590082e-03]]), atol=0.001) 
        np.testing.assert_allclose(ans2, np.array([-5.88784058, -5.88784058, 13.]), atol=0.001) 

        
    def test_plot(self):
       with warnings.catch_warnings():
           warnings.simplefilter("ignore")
           p = Lorenz(10, 8/3., 14)
           p.plot('lorenz.png')
            
           gold_image = cv2.imread('lorenz_gold.png')
           test_image = cv2.imread('lorenz.png')
            
           test_image_resized = skimage.transform.resize(test_image, 
                                                         (gold_image.shape[0], gold_image.shape[1]), 
                                                         mode='constant')
            
           ssim = skimage.measure.compare_ssim(skimage.img_as_float(gold_image), test_image_resized, multichannel=True)
           assert ssim >= 0.7

if __name__ == '__main__':
               unittest.main()
