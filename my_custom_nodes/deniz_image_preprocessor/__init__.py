from .image_preprocessor import ImagePreprocessor

NODE_CLASS_MAPPINGS = {
    "My Image Preprocessor": ImagePreprocessor,
}

__all__ = ["NODE_CLASS_MAPPINGS"]