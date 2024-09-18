# Advanced ControlNet Style Transfer

## Overview

`style_transfer.py` is a custom node for ComfyUI that demonstrates advanced style transfer capabilities using ControlNet. The node integrates ControlNet for enhanced style transfer results and provides fine-grained control over the style transfer process. Users can adjust parameters such as strength, start and end percentages, guidance scale, and control mode to achieve the desired style transfer effects. The node supports various input types, including conditioning, images, and control nets, making it flexible and customizable for different use cases.

## Features

- **Advanced ControlNet Integration**: Utilizes ControlNet for enhanced style transfer capabilities.
- **Fine-Grained Control**: Adjust parameters like strength, start and end percentages, guidance scale, and control mode.
- **Flexible Input Types**: Supports various input types including conditioning, images, and control nets.
- **Customizable**: Allows users to fine-tune the style transfer process to achieve desired results.


## Functions


### transfer_style

Main method for performing style transfer using ControlNet. It applies the specified style transfer parameters to the input images and control nets.

**Parameters:**
- `content_image` (torch.tensor): The content image tensor.
- `style_image` (torch.tensor): The style image tensor.
- `control_net` (torch.tensor): The control net tensor.
- `strength` (float): The strength of the style transfer.
- `start_pct` (float): The start percentage for the style transfer.
- `end_pct` (float): The end percentage for the style transfer.
- `guidance_scale` (float): The guidance scale for the style transfer.
- `control_mode` (str): The control mode for the style transfer.

**Returns:**
- `torch.tensor`: The stylized positive condition tensor.
- 'torch.tensor': The stylized negative condition tensor.


### apply_style_transfer

This function applies the style transfer process to the input images using the specified parameters. It leverages ControlNet to perform the style transfer and provides fine-grained control over the process.

**Parameters:**
- `conditioning` (list): A list of conditioning inputs to which style transfer will be applied.
- `style_fidelity` (float): The fidelity of the style transfer.

**Returns:**
- `list`: A list of styled conditioning inputs.


### apply_control_net

Built-in implementation of controlnet forward method. This function applies the ControlNet model to the provided conditioning inputs. It adjusts the conditioning inputs based on the specified parameters and integrates the ControlNet model into the process.

**Parameters:**
- `positive` (list): Conditioning input for positive guidance.
- `negative` (list): Conditioning input for negative guidance.
- `image` (torch.tensor): The input image to which ControlNet will be applied.
- `control_net` (torch.tensor): The ControlNet model to be used.
- `strength` (float): The strength of the ControlNet application.
- `start_percent` (float): The start percentage of the ControlNet application.
- `end_percent` (float): The end percentage of the ControlNet application.
- `vae` (torch.nn.Module): The Variational Autoencoder (VAE) model used in the process.

**Returns:**
- `tuple`: A tuple containing the modified positive and negative conditioning inputs.


### blend_conditioning

This function blends two sets of conditioning inputs based on a specified weight. It combines the conditioning inputs to create a blended output.

**Parameters:**
- `cond1` (list): The first set of conditioning inputs.
- `cond2` (list): The second set of conditioning inputs.
- `weight` (float): The blending weight.

**Returns:**
- `list`: A list of blended conditioning inputs.

### apply_guidance_scale

This function applies the guidance scale to the conditioning inputs. It scales the conditioning inputs based on the specified guidance scale.

**Parameters:**
- `conditioning` (list): The conditioning inputs to be scaled.
- `guidance_scale` (float): The guidance scale factor.

**Returns:**
- `list`: The scaled conditioning inputs.


### normalize_tensor

This function normalizes a tensor to maintain consistent magnitude. It ensures that the tensor values are within a specified range.

**Parameters:**
- `tensor` (torch.tensor): The tensor to be normalized.

**Returns:**
- `torch.tensor`: The normalized tensor.

**Description:**
The `normalize_tensor` method normalizes the input tensor to ensure that its values are within a specified range. This helps maintain consistent magnitude across different tensors and prevents any single tensor from dominating the style transfer process.











# Image Preprocessor

## Overview

`image_preprocessor.py` is a utility class for my custom preprocessor node. It provides functionality to validate, preprocess and resize images in order to be used in computer vision models. The preprocessor supports different techniques such as Canny edge detection and line art processing. The module leverages libraries like PyTorch, OpenCV, and NumPy to perform these operations efficiently.

## Features

- **Canny Edge Detection**: Apply Canny edge detection to highlight the edges in an image.
- **Line Art Processing**: Process images to extract line art.
- **Image Resizing with Padding**: Resize images to a specified size with padding to maintain aspect ratio.
- **Input Validation**: Validate input images to ensure they meet the required specifications.

# Image Preprocessor

## Overview

`image_preprocessor.py` is a Python module designed to preprocess images for various computer vision tasks. It provides functionality to apply different preprocessing techniques such as Canny edge detection and line art processing. The module leverages libraries like PyTorch, OpenCV, and NumPy to perform these operations efficiently.

## Features

- **Canny Edge Detection**: Apply Canny edge detection to highlight the edges in an image.
- **Line Art Processing**: Process images to extract line art.
- **Image Resizing with Padding**: Resize images to a specified size with padding to maintain aspect ratio.
- **Input Validation**: Validate input images to ensure they meet the required specifications.

## Functions

### preprocess_image

Main method for preprocessing images. It applies the specified preprocessor to the input image tensor.

**Parameters:**
- `image` (torch.tensor): The input image tensor to be preprocessed.
- `preprocessor` (str): The type of preprocessor to be applied (e.g., "CannyControlNet", "LineartControlNet").

**Returns:**
- `torch.tensor`: The preprocessed image tensor.

### input_validate

This function validates the input image to ensure it meets the required specifications. It checks the image's dimensions and type to ensure compatibility with the preprocessing functions.

**Parameters:**
- `image` (torch.tensor): The input image tensor to be validated.

**Returns:**
- `torch.tensor`: The validated image tensor.

### resize_image_with_pad

This function resizes the input image to a specified size with padding to maintain the aspect ratio. It ensures that the image fits within the target size without distortion.

**Parameters:**
- `image` (torch.tensor): The input image tensor to be resized.
- `target_size` (int): The target size for the resized image.

**Returns:**
- `torch.tensor`: The resized image tensor with padding.

### canny

This function applies Canny edge detection to the input image. Canny edge detection is a popular edge detection algorithm that highlights the edges in an image specifically useful for style transformation.

**Parameters:**
- `image` (torch.tensor): The input image tensor to which Canny edge detection will be applied.

**Returns:**
- `torch.tensor`: The image tensor with Canny edge detection applied.

### HWC3

This function converts the image to HWC (Height-Width-Channel) format. This format is commonly used in image processing tasks.

**Parameters:**
- `image` (torch.tensor): The input image tensor to be converted.

**Returns:**
- `torch.tensor`: The image tensor in HWC format.

### safer_memory

This function ensures the contiguity of the input tensor's memory layout. It is used to prevent memory errors during processing.

**Parameters:**
- `image` (torch.tensor): The input image tensor to be processed.

**Returns:**
- `torch.tensor`: The processed image tensor with contiguous memory layout.


### remove_pad

This function removes padding from the image. It is used to revert the image back to its original size after processing.

**Parameters:**
- `image` (torch.tensor): The input image tensor from which padding will be removed.

**Returns:**
- `torch.tensor`: The image tensor with padding removed.







