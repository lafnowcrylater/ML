import PIL.Image
import cv2
import numpy as np
import random
import concurrent.futures

def img_to_patches(input_image) -> tuple:
    """
    Returns 25x25 patches of a resized 200x200 image,
    for both grayscale and RGB color scales.

    Parameters:
    - input_image: Accepts an input path (str) or a file-like object.

    Returns:
    - grayscale_imgs: list of grayscale patches (as numpy arrays)
    - imgs: list of color patches (as numpy arrays)
    """
    # Open the image using PIL (supports both file paths and file-like objects)
    img = PIL.Image.open(input_image)
    
    # If a path is provided, check the extension; if not jpg/jpeg, convert to RGB
    if isinstance(input_image, str):
        ext = input_image.split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg']:
            img = img.convert('RGB')
    else:
        # For file-like objects, force conversion to RGB
        img = img.convert('RGB')
    
    # Resize the image to 200x200 if needed
    if img.size != (200, 200):
        img = img.resize((200, 200))
    
    patch_size = 25
    grayscale_imgs = []
    imgs = []
    
    # Generate patches by cropping the image into 25x25 blocks
    for i in range(0, img.height, patch_size):
        for j in range(0, img.width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            img_patch = img.crop(box)
            img_color = np.asarray(img_patch)
            grayscale_image = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
            grayscale_imgs.append(grayscale_image.astype(np.int32))
            imgs.append(img_color)
    
    return grayscale_imgs, imgs


def get_l1(v,x,y):
    l1=0
    # 1 to m, 1 to m-1
    for i in range(0,y-1):
        for j  in range(0,x):
            l1+=abs(v[j][i]-v[j][i+1])
    return l1

def get_l2(v,x,y):
    l2=0
    # 1 to m-1, 1 to m
    for i in range(0,y):
        for j  in range(0,x-1):
            l2+=abs(v[j][i]-v[j+1][i])
    return l2

def get_l3l4(v,x,y):
    l3=l4=0
    # 1 to m-1, 1 to m-1
    for i in range(0,y-1):
        for j  in range(0,x-1):
            l3+=abs(v[j][i]-v[j+1][i+1])
            l4+=abs(v[j+1][i]-v[j][i+1])

    return l3+l4

def get_pixel_var_degree_for_patch(patch:np.array)->int:
    """
    gives pixel variation for a given patch
    ---------------------------------------
    ## parameters:
    - patch: accepts a numpy array format of the patch of an image
    """
    x,y = patch.shape
    l1=l2=l3l4=0

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_l1 = executor.submit(get_l1,patch,x,y)
        future_l2 = executor.submit(get_l2,patch,x,y)
        future_l3l4 = executor.submit(get_l3l4,patch,x,y)

        l1 = future_l1.result()
        l2 = future_l2.result()
        l3l4 = future_l3l4.result()

    return  l1+l2+l3l4


def extract_rich_and_poor_textures(variance_values:list, patches:list):
    """
    returns a list of rich texture and poor texture patches respectively
    --------------------------------------------------------------------
    ## parameters:
    - variance_values: list of values that are pixel variances of each patch
    - color_patches: coloured patches of the target image
    """
    threshold = np.mean(variance_values)
    rich_texture_patches = []
    poor_texture_patches = []
    for i,j in enumerate(variance_values):
        if j >= threshold:
            rich_texture_patches.append(patches[i])
        else:
            poor_texture_patches.append(patches[i])

    return rich_texture_patches, poor_texture_patches


def get_complete_image(patches: list, coloured=True):
    # Check if patches list is empty
    if not patches:
        # Create a blank patch if the list is empty
        if coloured:
            blank_patch = np.zeros((25, 25, 3), dtype=np.float64)
        else:
            blank_patch = np.zeros((25, 25), dtype=np.float64)
        patches = [blank_patch] * 16
    else:
        random.shuffle(patches)
        # Handle the case where there aren't enough patches
        while len(patches) < 16:
            # Use modulo to avoid index errors when len(patches) is 1
            idx = random.randint(0, max(0, len(patches) - 1))
            patches.append(patches[idx])

    # Ensure we have exactly 16 patches (4x4 grid)
    if len(patches) > 16:
        patches = patches[:16]

    # For a 4x4 grid with 25x25 patches
    if coloured:
        grid = np.asarray(patches).reshape((4, 4, 25, 25, 3))
    else:
        grid = np.asarray(patches).reshape((4, 4, 25, 25))

    # Combine patches row-wise and then column-wise
    rows = [np.concatenate(grid[i, :], axis=1) for i in range(4)]
    img = np.concatenate(rows, axis=0)
    return img



def smash_n_reconstruct(input_image, coloured=True):
    """
    Performs the SmashnReconstruct part of preprocessing
    reference: [link](https://arxiv.org/abs/2311.12397)

    return rich_texture, poor_texture

    ----------------------------------------------------
    ## parameters:
    - input_image: Accepts input path of the image, a file-like object, or a PIL Image object
    """
    # Check if input is already a PIL Image
    if isinstance(input_image, PIL.Image.Image):
        image = input_image
    else:
        # Open the image if it's a path or file-like object
        image = PIL.Image.open(input_image)

    # Convert image to grayscale if needed
    if not coloured:
        image = image.convert('L')

    # Process the image to extract patches
    gray_scale_patches, color_patches = img_to_patches(image)

    pixel_var_degree = []
    for patch in gray_scale_patches:
        pixel_var_degree.append(get_pixel_var_degree_for_patch(patch))

    # r_patch = list of rich texture patches, p_patch = list of poor texture patches
    if coloured:
        r_patch, p_patch = extract_rich_and_poor_textures(variance_values=pixel_var_degree, patches=color_patches)
    else:
        r_patch, p_patch = extract_rich_and_poor_textures(variance_values=pixel_var_degree, patches=gray_scale_patches)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        rich_texture_future = executor.submit(get_complete_image, r_patch, coloured)
        poor_texture_future = executor.submit(get_complete_image, p_patch, coloured)

        rich_texture = rich_texture_future.result()
        poor_texture = poor_texture_future.result()

    return rich_texture, poor_texture