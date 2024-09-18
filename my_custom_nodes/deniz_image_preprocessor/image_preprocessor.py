import torch
import cv2
import numpy as np

class ImagePreprocessor:
    CATEGORY = "ImagePreprocessor"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "preprocessor": (
                    ["CannyControlNet", "LineartControlNet"],
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_image"

    def preprocess_image(self, image: torch.tensor, preprocessor: str) -> torch.tensor:

        image = self.input_validate(image)


        # Resize the image with padding
        image = self.resize_image_with_pad(image, 512)
  

        if preprocessor == "CannyControlNet":
            # Apply Canny edge detection
            print("Applying canny edge detection")
            image = self.canny(image)
            image = self.HWC3(self.remove_pad(image))
            print("Canny edge detection complete")

        elif preprocessor == "LineartControlNet":
            # Detect lineart in the image
            print("Applying lineart detection")
            image = self.lineart(image)
            image = self.HWC3(self.remove_pad(image))
            print("Lineart detection complete")
        
        else:
            raise ValueError(f"Invalid preprocessor: {preprocessor}")
        

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        
        image = torch.from_numpy(image.astype(np.uint8)).unsqueeze(0)
        print("Preprocessing complete")
        print(f"Image shape: {image.shape}")
        return (image,)

    def canny(
        self, image: torch.tensor, low_threshold: int = 100, high_threshold: int = 200
    ) -> torch.tensor:
        """Applies Canny edge detection to an image"""
        image = cv2.Canny(image, low_threshold, high_threshold)
        return image
    
    def lineart(self, image: torch.tensor,guassian_sigma=6.0, intensity_threshold=8) -> torch.tensor:
        """Detects lineart in an image"""
        intermediate = image.astype(np.float32)
        g = cv2.GaussianBlur(intermediate, (0, 0), guassian_sigma)
        intensity = np.min(g - intermediate, axis=2).clip(0, 255)
        intensity /= max(16, np.median(intensity[intensity > intensity_threshold]))
        intensity *= 127
        processed_image = intensity.clip(0, 255).astype(np.uint8)
        return processed_image

    def HWC3(self, x):
        """Converts an image to h,w,c format"""
        assert x.dtype == np.uint8 , "Image must be in uint8 format"

        if x.ndim == 2:
            x = x[:, :, None]

        elif x.ndim == 4:
            x = x.squeeze(0)

        assert x.ndim == 3
        H, W, C = x.shape
        assert C == 1 or C == 3 or C == 4
        if C == 3:
            return x
        if C == 1:
            return x
        if C == 4:
            color = x[:, :, 0:3].astype(np.float32)
            alpha = x[:, :, 3:4].astype(np.float32) / 255.0
            y = color * alpha + 255.0 * (1.0 - alpha)
            y = y.clip(0, 255).astype(np.uint8)

            return y

    def safer_memory(self, x) -> np.ndarray:
        """Returns a contiguous array of x to mitigate memory issues"""
        return np.ascontiguousarray(x.copy()).copy()

    def input_validate(self, input_image) -> np.ndarray:
        """ Validates the input image and converts it to a numpy array"""
        if input_image is None:
            raise ValueError("input_image must be defined.")

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.float32)

        
        
        input_image = (input_image * 255).astype("uint8")

        input_image = input_image.clip(0, 255)

        
        

        return input_image

    def resize_image_with_pad(self, input_image, resolution, mode="edge"):
        """Resizes an image to resolution with padding"""
        img = self.HWC3(input_image)

        H_raw, W_raw, _ = img.shape

        if resolution == 0:
            return img, lambda x: x
        
        
        k = float(resolution) / float(min(H_raw, W_raw))
        interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA

        self.H_target = int(np.round(float(H_raw) * k))
        self.W_target = int(np.round(float(W_raw) * k))
        img = cv2.resize(
            img,
            (self.W_target, self.H_target),
            interpolation=interpolation,
        )
        H_pad, W_pad = self.pad64(self.H_target), self.pad64(self.W_target)
        img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

        return img_padded

    def pad64(self, x):
        """Returns the padding required to make x a multiple of 64"""
        return int(np.ceil(float(x) / 64.0) * 64 - x)

    def remove_pad(self, x):
        """Removes the padding from x"""
        return self.safer_memory(x[: self.H_target, : self.W_target])
