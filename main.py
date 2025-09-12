import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import albumentations as A
from albumentations.pytorch import ToTensorV2

st.set_page_config(page_title="Image Augmentation Tool", page_icon="🖼️", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; padding: 10px 24px; border-radius: 4px; border: none;}
    .stButton>button:hover {background-color: #45a049;}
    .upload-section {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

def apply_augmentation(image, aug_type, params):
    img_array = np.array(image)
    transform = None

    if aug_type == "Rotation":
        transform = A.Compose([A.Rotate(limit=params['angle'], p=1.0), ToTensorV2()])
    elif aug_type == "Flip Horizontal":
        transform = A.Compose([A.HorizontalFlip(p=1.0), ToTensorV2()])
    elif aug_type == "Flip Vertical":
        transform = A.Compose([A.VerticalFlip(p=1.0), ToTensorV2()])
    elif aug_type == "Brightness":
        transform = A.Compose([A.RandomBrightnessContrast(brightness_limit=params['brightness'], p=1.0), ToTensorV2()])
    elif aug_type == "Blur":
        transform = A.Compose([A.GaussianBlur(blur_limit=params['blur'], p=1.0), ToTensorV2()])
    elif aug_type == "Noise":
        transform = A.Compose([A.GaussNoise(var_limit=params['noise'], p=1.0), ToTensorV2()])
    elif aug_type == "Shear":
        transform = A.Compose([A.Affine(shear=params['shear'], p=1.0), ToTensorV2()])
    elif aug_type == "Zoom":
        transform = A.Compose([A.RandomScale(scale_limit=params['zoom'], p=1.0), ToTensorV2()])
    elif aug_type == "Hue & Saturation":
        transform = A.Compose([A.HueSaturationValue(hue_shift_limit=params['hue'], sat_shift_limit=params['saturation'], val_shift_limit=0, p=1.0), ToTensorV2()])
    elif aug_type == "Random Perspective":
        transform = A.Compose([A.RandomPerspective(scale=params['perspective'], p=1.0), ToTensorV2()])

    augmented = transform(image=img_array)
    aug_img = augmented['image']
    if isinstance(aug_img, np.ndarray):
        return Image.fromarray(aug_img)
    else:
        return Image.fromarray(aug_img.numpy().transpose(1,2,0))

def main():
    st.title("🖼️ Image Augmentation Tool")
    st.write("Upload an image and apply various augmentations for training or experimentation!")

    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        st.sidebar.header("Augmentation Options")
        aug_type = st.sidebar.selectbox("Select Augmentation Type", [
            "Rotation", "Flip Horizontal", "Flip Vertical", "Brightness", "Blur", "Noise", "Shear", "Zoom", "Hue & Saturation", "Random Perspective"
        ])

        params = {}
        if aug_type == "Rotation":
            params['angle'] = st.sidebar.slider("Rotation Angle", -180, 180, 0)
        elif aug_type == "Brightness":
            params['brightness'] = st.sidebar.slider("Brightness", 0.0, 2.0, 1.0)
        elif aug_type == "Blur":
            params['blur'] = st.sidebar.slider("Blur Intensity", 1, 15, 3)
        elif aug_type == "Noise":
            params['noise'] = st.sidebar.slider("Noise Level", 0, 100, 10)
        elif aug_type == "Shear":
            params['shear'] = st.sidebar.slider("Shear Angle", -45, 45, 0)
        elif aug_type == "Zoom":
            params['zoom'] = st.sidebar.slider("Zoom Factor", -50, 50, 0)
        elif aug_type == "Hue & Saturation":
            params['hue'] = st.sidebar.slider("Hue Shift", -50, 50, 0)
            params['saturation'] = st.sidebar.slider("Saturation Shift", -50, 50, 0)
        elif aug_type == "Random Perspective":
            params['perspective'] = st.sidebar.slider("Perspective Scale", 0.0, 0.5, 0.2)

        if st.button("Apply Augmentation"):
            with st.spinner("Applying augmentation..."):
                augmented_image = apply_augmentation(image, aug_type, params)
                st.image(augmented_image, caption=f"Augmented Image ({aug_type})", use_column_width=True)

                img_byte_arr = io.BytesIO()
                augmented_image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                st.download_button(
                    label="Download Augmented Image",
                    data=img_byte_arr,
                    file_name=f"augmented_{aug_type.replace(' ', '_').lower()}.png",
                    mime="image/png"
                )

    st.sidebar.header("About")
    st.sidebar.info("""
    This tool helps prepare training data with advanced image augmentations.
    Available Augmentations:
    - Rotation, Flip, Brightness, Blur, Noise
    - Shear, Zoom, Hue & Saturation, Random Perspective
    """)

if __name__ == "__main__":
    main()
