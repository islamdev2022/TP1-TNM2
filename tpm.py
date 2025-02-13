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
        
        ctk.CTkLabel(mode_frame, text="Sampling Mode:").pack()
        
        self.mode_var = ctk.StringVar(value="4:4:4")
        modes = ["4:4:4", "4:4:2", "4:2:2"]
        for mode in modes:
            ctk.CTkRadioButton(
                mode_frame, 
                text=mode, 
                variable=self.mode_var, 
                value=mode
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
        """
        Convert RGB image to YCrCb color space
        Uses standard conversion matrix for color space transformation
        """
        transform_matrix = np.array([
            [0.299, 0.587, 0.114],      # Y component
            [-0.168736, -0.331264, 0.5], # Cb component
            [0.5, -0.418688, -0.081312]  # Cr component
        ])

        img_array = np.array(image, dtype=np.float32)
        R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        # Calculate YCrCb components
        Y = transform_matrix[0, 0] * R + transform_matrix[0, 1] * G + transform_matrix[0, 2] * B
        Cb = transform_matrix[1, 0] * R + transform_matrix[1, 1] * G + transform_matrix[1, 2] * B + 128
        Cr = transform_matrix[2, 0] * R + transform_matrix[2, 1] * G + transform_matrix[2, 2] * B + 128

        return np.stack((Y, Cb, Cr), axis=-1).astype(np.uint8)

    def downsample(self, ycrcb_image, mode):
        """
        Perform chroma subsampling based on selected mode
        4:4:4 - No subsampling
        4:4:2 - Vertical subsampling
        4:2:2 - Horizontal subsampling
        """
        Y, Cb, Cr = ycrcb_image[:, :, 0], ycrcb_image[:, :, 1], ycrcb_image[:, :, 2]
        if mode == "4:4:2":
            Cb = Cb[::2, :]  # Vertical subsampling
            Cr = Cr[::2, :]
        elif mode == "4:2:2":
            Cb = Cb[:, ::2]  # Horizontal subsampling
            Cr = Cr[:, ::2]

        return Y, Cb, Cr

    def psnr(self, original, compressed):
        """Calculate Peak Signal-to-Noise Ratio between original and compressed images"""
        mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
        return 10 * np.log10(255 ** 2 / mse) if mse != 0 else float('inf')

    def show_images(self, original, transformed, title):
        """Display original and processed images side by side"""
        plt.close('all')  # Close any existing plots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(original)
        axes[0].set_title("Original Image (RGB)")
        axes[0].axis("off")

        axes[1].imshow(transformed, cmap='gray' if len(transformed.shape) == 2 else None)
        axes[1].set_title(title)
        axes[1].axis("off")

        plt.show()

    def process_image(self):
        """Process the selected image with the chosen sampling mode"""
        if not self.image_path:
            return

        # Load and convert image
        self.original_image = Image.open(self.image_path).convert("RGB")
        ycrcb_image = self.rgb_to_ycrcb(self.original_image)

        mode = self.mode_var.get()
        original_size = self.original_image.size[0] * self.original_image.size[1] * 3

        # Process image based on selected mode
        if mode == "4:4:4":
            self.processed_image = ycrcb_image
            processed_size = original_size
        else:
            Y, Cb, Cr = self.downsample(ycrcb_image, mode)
            processed_size = Y.size + Cb.size + Cr.size
            self.processed_image = np.stack((
                Y,
                np.repeat(Cb, 2, axis=0) if mode == "4:4:2" else np.repeat(Cb, 2, axis=1),
                np.repeat(Cr, 2, axis=0) if mode == "4:4:2" else np.repeat(Cr, 2, axis=1)
            ), axis=-1)

        # Calculate and display metrics
        conversion_rate = original_size / processed_size
        psnr_value = self.psnr(ycrcb_image, self.processed_image)

        self.size_label.configure(
            text=f"Original size: {original_size} pixels\nProcessed size: {processed_size} pixels"
        )
        self.conv_label.configure(text=f"Conversion rate: {conversion_rate:.2f}")
        self.psnr_label.configure(text=f"PSNR: {psnr_value:.2f} dB")

        # Show images and enable save button
        self.show_images(self.original_image, self.processed_image, f"YCrCb Image ({mode})")
        self.save_button.configure(state="normal")

    def save_image(self):
        """Save the processed image"""
        if self.processed_image is not None:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".bmp",
                filetypes=[("BMP files", "*.bmp")]
            )
            if save_path:
                Image.fromarray(self.processed_image.astype(np.uint8)).save(save_path)

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageProcessorGUI()
    app.run()