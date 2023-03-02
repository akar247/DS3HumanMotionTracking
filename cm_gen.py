import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel('practice_imgs.xlsx')

def plot_image(data, title = "cms"):
    """Plots image num in the list
    
    Args:
        data: The array where the image data is located
        img_num: The sample number in the dataset

    Returns:
        Plot of the image
    """
    fig = plt.figure
    plt.title(title)
    plt.imshow(data, cmap='gray')
    
    
    return plt.show() 

def make_sampling_vector(width, height, stride):
    """Create sampling vectors that define a grid.

    Args:
        width: Width of grid as an integer.
        height: Height of grid as an integer.
        stride: Steps between each sample.
    
    Returns:
        A pair of vectors xv, yv that define the grid.
    """
    xv = np.arange(0, width, stride, dtype="float32")
    yv = np.arange(0, height, stride, dtype="float32")
    return xv, yv


def make_confidence_map(x, y, xv, yv, sigma=1):
    """Make confidence maps for a point.

    Args:
        x: X-coordinate of the center of the confidence map.
        y: Y-coordinate of the center of the confidence map.
        xv: X-sampling vector.
        yv: Y-sampling vector.
        sigma: Spread of the confidence map.
    
    Returns:
        A confidence map centered at the x, y coordinates specified as
        a 2D array of shape (len(yv), len(xv)).
    """

    cm = np.exp(-(
    (xv.reshape(1, -1) - x) ** 2 + (yv.reshape(-1, 1) - y) ** 2) / (2* sigma ** 2))
    return cm

  

def make_multi_nodal_cm(cord_array, img_height, img_width, stride, sigma):
    """Creates a confidence maps for n-nodes

    Args:
      cord_array: array containing elements of xy pairs
      img_height: height of input image
      img_width: width of input image

    Returns:
      Confidence maps aggregated nodes
    """

    output_cms = []

    for cord in cord_array:
      xv, yv = make_sampling_vector(img_width, img_height, stride)
      cord_cm = make_confidence_map(cord[0], cord[1], xv, yv, sigma)
      output_cms.append(cord_cm)

    return output_cms



data = pd.read_csv('practice_imgs.csv')
coords = data.reset_index(drop=True).iloc[:, 1:-3]

coords_comb = pd.DataFrame()

for i in range(0, len(coords.columns), 2):
     name = list(coords.columns)[i].split('_')[0]
     coords_comb[name] = list(zip(coords.iloc[:,i], coords.iloc[:,i+1])) 

# for col in coords_comb.columns:
#     coords_comb[col] = pd.to_numeric(coords_comb[col], errors='coerce') 

cms = make_multi_nodal_cm(coords_comb.iloc[0], 1000, 1000, 2, 10)

#plot_image(cms[i])