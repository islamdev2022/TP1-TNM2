import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog
import os

class ImageProcessorGUI:
    def __init__(self):
        # Configure the appearance of customtkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Create main window
        self.root = ctk.CTk()
        self.root.title("Image Color Space Converter")
        self.root.geometry("800x600")

        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.Y = None
        self.Cb = None
        self.Cr = None

        self.setup_gui()

    def setup_gui(self):
        # Create main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # File selection button
        self.select_button = ctk.CTkButton(
            main_frame, 
            text="Select Image", 
            command=self.select_image
        )
        self.select_button.pack(pady=10)

        # Display selected file path
        self.file_label = ctk.CTkLabel(main_frame, text="No file selected")
        self.file_label.pack(pady=5)

        # Sampling mode selection
        mode_frame = ctk.CTkFrame(main_frame)
        mode_frame.pack(pady=10)
        
        ctk.CTkLabel(mode_frame, text="Échantillonnage:").pack()
        
        self.mode_var = ctk.StringVar(value="1")
        modes = [("4:4:4", "1"), ("4:2:2", "2"), ("4:2:0", "3")]
        for text, value in modes:
            ctk.CTkRadioButton(
                mode_frame, 
                text=f"{value}={text}", 
                variable=self.mode_var, 
                value=value
            ).pack(pady=5)

        # Process button
        self.process_button = ctk.CTkButton(
            main_frame,
            text="Process Image",
            command=self.process_image,
            state="disabled"
        )
        self.process_button.pack(pady=10)

        # Results frame
        self.results_frame = ctk.CTkFrame(main_frame)
        self.results_frame.pack(pady=10, fill="x")

        # Labels for displaying metrics
        self.size_label = ctk.CTkLabel(self.results_frame, text="")
        self.size_label.pack()
        
        self.conv_label = ctk.CTkLabel(self.results_frame, text="")
        self.conv_label.pack()
        
        self.psnr_label = ctk.CTkLabel(self.results_frame, text="")
        self.psnr_label.pack()

        # Save button
        self.save_button = ctk.CTkButton(
            main_frame,
            text="Save Processed Image",
            command=self.save_image,
            state="disabled"
        )
        self.save_button.pack(pady=10)

    def select_image(self):
        """Handle image file selection"""
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.bmp *.png *.jpg *.jpeg")]
        )
        if self.image_path:
            self.file_label.configure(text=os.path.basename(self.image_path))
            self.process_button.configure(state="normal")

    def rgb_to_ycrcb(self, image):
        """Convert RGB image to YCrCb color space"""
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

        return Y, Cb, Cr

    def downsample(self, Y, Cb, Cr, mode):
        """Downsample Cb and Cr components based on mode"""
        if mode == "2":  # 4:2:2
            Cb = Cb[:, ::2]
            Cr = Cr[:, ::2]
        elif mode == "3":  # 4:2:0
            Cb = Cb[::2, ::2]
            Cr = Cr[::2, ::2]
        return Y, Cb, Cr

    def upsample(self, Cb, Cr, mode):
        """Upsample Cb and Cr components back to original size"""
        if mode == "2":  # 4:2:2
            Cb = np.repeat(Cb, 2, axis=1)
            Cr = np.repeat(Cr, 2, axis=1)
        elif mode == "3":  # 4:2:0
            Cb = np.repeat(np.repeat(Cb, 2, axis=0), 2, axis=1)
            Cr = np.repeat(np.repeat(Cr, 2, axis=0), 2, axis=1)
        return Cb, Cr

    def ycrcb_to_rgb(self, Y, Cb, Cr):
        """Convert YCrCb back to RGB color space"""
        Cb_temp = Cb - 128
        Cr_temp = Cr - 128
        
        inv_transform = np.array([
            [1, 0, 1.402],
            [1, -0.344136, -0.714136],
            [1, 1.772, 0]
        ])

        R = inv_transform[0, 0] * Y + inv_transform[0, 2] * Cr_temp
        G = inv_transform[1, 0] * Y + inv_transform[1, 1] * Cb_temp + inv_transform[1, 2] * Cr_temp
        B = inv_transform[2, 0] * Y + inv_transform[2, 1] * Cb_temp

        rgb_image = np.clip(np.stack((R, G, B), axis=-1), 0, 255).astype(np.uint8)
        return rgb_image

    def psnr(self, original, compressed):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
        return 10 * np.log10(255 ** 2 / mse) if mse != 0 else float('inf')

    def process_image(self):
        """Process the selected image with the chosen sampling mode"""
        if not self.image_path:
            return

        # Load and convert image
        self.original_image = Image.open(self.image_path).convert("RGB")
        Y, Cb, Cr = self.rgb_to_ycrcb(self.original_image)
        
        mode = self.mode_var.get()
        original_size = self.original_image.size[0] * self.original_image.size[1] * 3

        # Downsample
        Y_down, Cb_down, Cr_down = self.downsample(Y, Cb, Cr, mode)
        compressed_size = Y_down.size + Cb_down.size + Cr_down.size

        # Upsample
        Cb_up, Cr_up = self.upsample(Cb_down, Cr_down, mode)
        
        # Reconstruct RGB
        self.processed_image = self.ycrcb_to_rgb(Y_down, Cb_up, Cr_up)
        
        # Store YCbCr components
        self.Y = Y
        self.Cb = Cb
        self.Cr = Cr

        # Calculate metrics
        psnr_value = self.psnr(np.array(self.original_image), self.processed_image)
        conversion_rate = original_size / compressed_size if mode != "1" else 1

        # Update labels
        self.size_label.configure(
            text=f"Original size: {original_size} pixels\nProcessed size: {compressed_size} pixels"
        )
        self.conv_label.configure(text=f"Conversion rate: {conversion_rate:.2f}")
        self.psnr_label.configure(text=f"PSNR: {psnr_value:.2f} dB")

        # Display images
        self.save_button.configure(state="normal")
        self.display_images()

    def display_images(self):
        """Display all images in the required format"""
        plt.close('all')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Display images in first row
        axes[0, 0].imshow(self.original_image)
        axes[0, 0].set_title("Image Originale (RGB)")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(self.processed_image)
        axes[0, 1].set_title("Image RGB Reconstituée")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(self.Cb, cmap='gray')
        axes[0, 2].set_title("Canal Cb")
        axes[0, 2].axis("off")

        # Display images in second row
        axes[1, 0].imshow(self.Cr, cmap='gray')
        axes[1, 0].set_title("Canal Cr")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(self.Y, cmap='gray')
        axes[1, 1].set_title("Canal Y")
        axes[1, 1].axis("off")

        # Remove the unused subplot
        fig.delaxes(axes[1, 2])

        plt.tight_layout()
        plt.show()

    def save_image(self):
        """Save the processed image"""
        if self.processed_image is not None:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".bmp",
                filetypes=[("BMP files", "*.bmp")]
            )
            if save_path:
                Image.fromarray(self.processed_image).save(save_path)

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageProcessorGUI()
    app.run()