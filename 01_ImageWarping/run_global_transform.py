import gradio as gr
import cv2
import numpy as np

import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）

    im_size = image.shape[:2]
    im_center = (im_size[0]//2, im_size[1]//2)

    scale_matrix = np.array([[scale, 0, (1.-scale)*im_center[0]], [0, scale, (1.-scale)*im_center[1]], [0, 0, 1]])
    cosa = np.cos(np.radians(rotation))
    sina = np.sin(np.radians(rotation))
    # rotation_matrix = np.array([[np.cos(np.radians(rotation)), -np.sin(np.radians(rotation)), 0], [np.sin(np.radians(rotation)), np.cos(np.radians(rotation)), 0], [0, 0, 1]])
    rotation_matrix = np.array([[cosa, -sina, (1.-cosa)*im_center[0]+sina*im_center[1]], [sina, cosa, -sina*im_center[0]+(1.-cosa)*im_center[1]], [0, 0, 1]])
    translation_matrix = np.array([[0, 0, translation_x], [0, 0, translation_y], [0, 0, 1]])

    transformed_matrix = scale_matrix @ rotation_matrix + translation_matrix
    transformed_image = cv2.warpAffine(transformed_image, to_3x3(transformed_matrix)[:2], (transformed_image.shape[1], transformed_image.shape[0]))

    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
