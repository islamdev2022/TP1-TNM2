import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def rgb_to_ycrcb(image):

    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])

    img_array = np.array(image, dtype=np.float32)
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    Y = transform_matrix[0, 0] * R + transform_matrix[0, 1] * G + transform_matrix[0, 2] * B
    Cb = transform_matrix[1, 0] * R + transform_matrix[1, 1] * G + transform_matrix[1, 2] * B + 128
    Cr = transform_matrix[2, 0] * R + transform_matrix[2, 1] * G + transform_matrix[2, 2] * B + 128

    return np.stack((Y, Cb, Cr), axis=-1).astype(np.uint8)


def downsample(ycrcb_image, mode):
   
    Y, Cb, Cr = ycrcb_image[:, :, 0], ycrcb_image[:, :, 1], ycrcb_image[:, :, 2]
    if mode == "4:4:2":
        Cb = Cb[::2, :]
        Cr = Cr[::2, :]
    elif mode == "4:2:2":
        Cb = Cb[:, ::2]
        Cr = Cr[:, ::2]

    return Y, Cb, Cr


def psnr(original, compressed):
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    return 10 * np.log10(255 ** 2 / mse) if mse != 0 else float('inf')


def show_images(original, transformed, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Image Originale (RGB)")
    axes[0].axis("off")

    axes[1].imshow(transformed, cmap='gray' if len(transformed.shape) == 2 else None)
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.show()


def main():
    import os
    image_path = os.path.join("C:", "Users", "sts", "Pictures", "fa.bmp")
    image = Image.open(image_path).convert("RGB")
    ycrcb_image = rgb_to_ycrcb(image)

    mode_choice = input("Choisissez l'échantillonnage (1 = 4:4:4, 2 = 4:4:2, 3 = 4:2:2) : ")
    mode_mapping = {"1": "4:4:4", "2": "4:4:2", "3": "4:2:2"}
    mode = mode_mapping.get(mode_choice, "4:4:4")

    original_size = image.size[0] * image.size[1] * 3  # Taille de l'image RGB

    if mode == "4:4:4":
        processed_image = ycrcb_image
        processed_size = original_size
    else:
        Y, Cb, Cr = downsample(ycrcb_image, mode)
        processed_size = Y.size + Cb.size + Cr.size
        processed_image = np.stack((Y, np.repeat(Cb, 2, axis=0) if mode == "4:4:2" else np.repeat(Cb, 2, axis=1),
                                    np.repeat(Cr, 2, axis=0) if mode == "4:4:2" else np.repeat(Cr, 2, axis=1)), axis=-1)

    conversion_rate = original_size / processed_size

    psnr_value = psnr(ycrcb_image, processed_image)

    print(f"Taille de l'image originale : {original_size} pixels")
    print(f"Taille de l'image transformée : {processed_size} pixels")
    print(f"Taux de conversion : {conversion_rate:.2f}")
    print(f"PSNR : {psnr_value:.2f} dB")

    show_images(image, processed_image, f"Image YCrCb ({mode})")

    save_option = input("Voulez-vous enregistrer l'image transformée ? (oui/non) : ")
    if save_option.lower() == "oui":
        Image.fromarray(processed_image.astype(np.uint8)).save("output.bmp")
        print("Image enregistrée sous 'output.bmp'")


if __name__ == "__main__":
    main()
