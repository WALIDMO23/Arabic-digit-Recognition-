import os
import time
import warnings
import numpy as np
import cv2
import joblib
from PIL import Image, ImageOps, ImageTk, ImageDraw
from skimage.feature import hog, local_binary_pattern
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

warnings.filterwarnings("ignore")


class DigitRecognizer:
    def __init__(self, model_path="models/best_model.pkl",
                 scaler_path="models/scaler.pkl",
                 info_path="models/model_info.pkl"):
        self.arabic_digits = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]
        self.english_names = ["Zero", "One", "Two", "Three", "Four",
                              "Five", "Six", "Seven", "Eight", "Nine"]

        self.model = None
        self.scaler = None
        self.model_info = {"feature_config": {"img_size": (64, 64),
                                              "use_hog": True,
                                              "use_lbp": True,
                                              "hog_orientations": 9},
                           "algorithm": "Unknown",
                           "test_accuracy": 0.0}

        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                raise FileNotFoundError(model_path)

            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                self.scaler = None

            if os.path.exists(info_path):
                self.model_info = joblib.load(info_path)

            print(f"✅ Model loaded: {self.model_info.get('algorithm', 'Unknown')}")
            print(f"   Test Accuracy: {self.model_info.get('test_accuracy', 0.0):.4f}")

        except Exception as e:
            print(f"❌ Error loading model artifacts: {e}")
            try:
                messagebox.showerror("Error", f"Failed to load model artifacts: {e}")
            except Exception:
                pass
            self.model = None


    def preprocess_image(self, image_path):
        try:
            img = Image.open(image_path).convert("L")
            img_array = np.array(img)

            if np.mean(img_array) < 128:
                img = ImageOps.invert(img)
                img_array = np.array(img)

            img_size = tuple(self.model_info.get("feature_config", {}).get("img_size", (64, 64)))
            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                resample = Image.LANCZOS
            img = img.resize(img_size, resample)
            img_array = np.array(img)

            return img_array

        except Exception as e:
            print(f"❌ Error loading/preprocessing image '{image_path}': {e}")
            return None


    def extract_features(self, img_array):
        if img_array is None:
            return None

        features = []
        config = self.model_info.get("feature_config", {})

        if img_array.ndim == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_array

        if config.get("use_hog", True):
            orientations = int(config.get("hog_orientations", 9))
            hog_feat = hog(img_gray,
                           orientations=orientations,
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           channel_axis=None,
                           feature_vector=True)
            features.extend(hog_feat.tolist())

        if config.get("use_lbp", True):
            P = int(config.get("lbp_points", 24))
            R = int(config.get("lbp_radius", 3))
            lbp = local_binary_pattern(img_gray, P, R, method="uniform")
            bins = P + 2
            hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
            hist = hist.astype("float32")
            if hist.sum() > 0:
                hist /= (hist.sum() + 1e-9)
            features.extend(hist.tolist())

        if len(features) == 0:
            return None

        return np.array(features, dtype="float32")


    def predict(self, image_path):
        if self.model is None:
            return None

        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return None

        feats = self.extract_features(img_array)
        if feats is None:
            return None

        X = np.array([feats])
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                print(f"Warning: scaler transform failed: {e}")

        try:
            pred = int(self.model.predict(X)[0])
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

        probs = None
        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0].tolist()
            else:
                probs = [0.0] * 10
                probs[pred] = 1.0
        except Exception:
            probs = [0.0] * 10
            probs[pred] = 1.0

        confidence = float(probs[pred] * 100.0)

        return {
            "prediction": pred,
            "arabic_digit": self.arabic_digits[pred] if 0 <= pred < 10 else "?",
            "english_name": self.english_names[pred] if 0 <= pred < 10 else "Unknown",
            "confidence": confidence,
            "probabilities": np.array(probs),
            "image_array": img_array
        }


    def predict_from_array(self, img_array):
        if self.model is None:
            return None

        if img_array is None:
            return None

        img_size = tuple(self.model_info.get("feature_config", {}).get("img_size", (64, 64)))
        try:
            pil_img = Image.fromarray(img_array.astype("uint8"))
        except Exception:
            pil_img = Image.fromarray((img_array).astype("uint8"))

        try:
            resample = Image.Resampling.LANCZOS
        except Exception:
            resample = Image.LANCZOS
        pil_img = pil_img.resize(img_size, resample)
        arr = np.array(pil_img)

        if np.mean(arr) < 128:
            arr = 255 - arr

        return self.predict_array_internal(arr)


    def predict_array_internal(self, arr):
        feats = self.extract_features(arr)
        if feats is None:
            return None
        X = np.array([feats])
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                print(f"Warning: scaler transform failed: {e}")

        try:
            pred = int(self.model.predict(X)[0])
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0].tolist()
            else:
                probs = [0.0] * 10
                probs[pred] = 1.0
        except Exception:
            probs = [0.0] * 10
            probs[pred] = 1.0

        confidence = float(probs[pred] * 100.0)
        return {
            "prediction": pred,
            "arabic_digit": self.arabic_digits[pred] if 0 <= pred < 10 else "?",
            "english_name": self.english_names[pred] if 0 <= pred < 10 else "Unknown",
            "confidence": confidence,
            "probabilities": np.array(probs),
            "image_array": arr
        }


class DrawingCanvas:
    def __init__(self, parent, width=280, height=280):
        self.parent = parent
        self.width = width
        self.height = height

        self.canvas = tk.Canvas(parent, width=width, height=height,
                                bg="white", cursor="cross")
        self.canvas.pack()

        self.image = Image.new("L", (width, height), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.brush_size = 20
        self.last_x = None
        self.last_y = None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)


    def paint(self, event):
        x, y = event.x, event.y

        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=self.brush_size,
                                    fill="black",
                                    capstyle=tk.ROUND,
                                    smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill=0, width=self.brush_size)

        self.last_x = x
        self.last_y = y


    def reset(self, event):
        self.last_x = None
        self.last_y = None


    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.width, self.height), 255)
        self.draw = ImageDraw.Draw(self.image)


    def get_image_array(self):
        try:
            resample = Image.Resampling.LANCZOS
        except Exception:
            resample = Image.LANCZOS
        img_resized = self.image.resize((64, 64), resample)
        return np.array(img_resized)


    def get_image_for_display(self):
        return self.image.copy()


class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Arabic Digit Recognition - Draw & Predict")
        self.root.geometry("1000x700")

        self.recognizer = DigitRecognizer()

        self.bg_color = "#f0f0f0"
        self.primary_color = "#2c3e50"
        self.secondary_color = "#3498db"
        self.accent_color = "#e74c3c"
        self.draw_color = "#27ae60"

        self.root.configure(bg=self.bg_color)

        self.current_image_path = None
        self.upload_display_img = None  # keep reference to PhotoImage to avoid GC
        self.drawing_canvas_widget = None

        self.upload_result = None
        self.draw_result = None

        self.setup_ui()


    def setup_ui(self):
        title_frame = tk.Frame(self.root, bg=self.primary_color, height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title_label = tk.Label(title_frame,
                               text="🔢 Arabic Digit Recognition - Draw & Predict",
                               font=("Arial", 18, "bold"),
                               bg=self.primary_color,
                               fg="white")
        title_label.pack(expand=True)

        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        left_frame = tk.Frame(main_frame, bg=self.bg_color)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_upload = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_upload, text="📁 Upload Image")

        tk.Label(self.tab_upload,
                 text="Select Image File",
                 font=("Arial", 12, "bold"),
                 bg="white").pack(pady=10)

        self.upload_canvas_frame = tk.Frame(self.tab_upload, bg="white")
        self.upload_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.upload_canvas = tk.Canvas(self.upload_canvas_frame, bg="white",
                                       highlightthickness=1, highlightbackground="#ddd")
        self.upload_canvas.pack(fill=tk.BOTH, expand=True)

        # Buttons for upload tab: Browse and Clear (Clear helps re-use)
        btn_frame = tk.Frame(self.tab_upload, bg="white")
        btn_frame.pack(pady=10)

        self.btn_upload = tk.Button(btn_frame,
                                    text="📁 Browse Image",
                                    command=self.load_image,
                                    bg=self.secondary_color,
                                    fg="white",
                                    font=("Arial", 11),
                                    height=2,
                                    width=20)
        self.btn_upload.pack(side=tk.LEFT, padx=5)

        self.btn_clear_upload = tk.Button(btn_frame,
                                          text="🧹 Clear Upload",
                                          command=self.clear_uploaded_image,
                                          bg="#e74c3c",
                                          fg="white",
                                          font=("Arial", 11),
                                          height=2,
                                          width=14)
        self.btn_clear_upload.pack(side=tk.LEFT, padx=5)

        self.tab_draw = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_draw, text="✏️ Draw Digit")

        tk.Label(self.tab_draw,
                 text="Draw a digit (0-9) below",
                 font=("Arial", 12, "bold"),
                 bg="white").pack(pady=10)

        tk.Label(self.tab_draw,
                 text="Tip: Draw large and centered for best results",
                 font=("Arial", 9),
                 fg="#7f8c8d",
                 bg="white").pack()

        self.drawing_canvas_widget = DrawingCanvas(self.tab_draw, width=280, height=280)

        btn_clear = tk.Button(self.tab_draw, text="🧹 Clear Drawing", command=self.drawing_canvas_widget.clear_canvas,
                              bg="#e74c3c", fg="white", font=("Arial", 10))
        btn_clear.pack(pady=8)

        self.btn_predict_drawing = tk.Button(self.tab_draw,
                                             text="🔍 Predict Drawing",
                                             command=self.predict_drawing,
                                             bg=self.draw_color,
                                             fg="white",
                                             font=("Arial", 11),
                                             height=2,
                                             width=20)
        self.btn_predict_drawing.pack(pady=8)

        right_frame = tk.Frame(main_frame, bg=self.bg_color)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        results_frame = tk.LabelFrame(right_frame,
                                      text="📊 Prediction Results",
                                      font=("Arial", 14, "bold"),
                                      bg="white",
                                      relief=tk.GROOVE,
                                      borderwidth=2)
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.result_display = tk.Frame(results_frame, bg="white")
        self.result_display.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.initial_message = tk.Label(self.result_display,
                                        text="Load or draw a digit to see predictions",
                                        font=("Arial", 12),
                                        fg="#7f8c8d",
                                        bg="white")
        self.initial_message.pack(expand=True)

        info_frame = tk.LabelFrame(right_frame,
                                   text="ℹ️ Model Information",
                                   font=("Arial", 11),
                                   bg="white",
                                   relief=tk.GROOVE,
                                   borderwidth=1)
        info_frame.pack(fill=tk.X, pady=(20, 0))

        if self.recognizer.model_info:
            info_text = f"Algorithm: {self.recognizer.model_info.get('algorithm', 'Unknown')}\n"
            info_text += f"Test Accuracy: {self.recognizer.model_info.get('test_accuracy', 0.0):.2%}\n"
            info_text += f"Training Time: {self.recognizer.model_info.get('training_time', 'N/A')}s"

            tk.Label(info_frame,
                     text=info_text,
                     font=("Arial", 9),
                     bg="white",
                     justify=tk.LEFT).pack(padx=10, pady=10)

        action_frame = tk.Frame(right_frame, bg=self.bg_color)
        action_frame.pack(fill=tk.X, pady=(20, 0))

        self.btn_predict_upload = tk.Button(action_frame,
                                            text="🔍 Predict Uploaded Image",
                                            command=self.predict_uploaded_image,
                                            bg="#27ae60",
                                            fg="white",
                                            font=("Arial", 11),
                                            height=2)
        self.btn_predict_upload.pack(fill=tk.X, pady=5)

        self.btn_compare = tk.Button(action_frame,
                                     text="🔄 Compare Results",
                                     command=self.compare_results,
                                     bg="#9b59b6",
                                     fg="white",
                                     font=("Arial", 11),
                                     height=2)
        self.btn_compare.pack(fill=tk.X, pady=5)

        self.status_bar = tk.Label(self.root,
                                   text="Welcome! Upload an image or draw a digit to begin.",
                                   bd=1,
                                   relief=tk.SUNKEN,
                                   anchor=tk.W,
                                   bg="#ecf0f1",
                                   fg="#2c3e50",
                                   font=("Arial", 9))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)


    # -------------------------
    # Upload helpers / fixes
    # -------------------------
    def clear_uploaded_image(self):
        """Clear the uploaded image and related state so user can upload another image."""
        self.current_image_path = None
        self.upload_display_img = None
        self.upload_canvas.delete("all")
        # Clear any previous upload result but keep draw_result intact
        self.upload_result = None
        # Reset result display to initial message if no draw result exists
        if not self.draw_result:
            for widget in self.result_display.winfo_children():
                widget.destroy()
            self.initial_message = tk.Label(self.result_display,
                                            text="Load or draw a digit to see predictions",
                                            font=("Arial", 12),
                                            fg="#7f8c8d",
                                            bg="white")
            self.initial_message.pack(expand=True)
        self.status_bar.config(text="Upload cleared. You can choose a new image.")


    def load_image(self):
        """
        Open file dialog and display the selected image.
        Important: this function resets upload_result and UI so multiple uploads
        work without restarting the program.
        """
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.gif"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            # Reset upload-related state (allow repeated upload/predict)
            self.current_image_path = file_path
            self.upload_result = None
            # Clear previous display (avoid stacking PhotoImage objects)
            self.upload_canvas.delete("all")
            # Display the newly selected image
            self.display_uploaded_image(file_path)
            # Clear result panel (so previous result doesn't stay)
            for widget in self.result_display.winfo_children():
                widget.destroy()
            self.initial_message = tk.Label(self.result_display,
                                            text="Ready: press 'Predict Uploaded Image' to classify.",
                                            font=("Arial", 12),
                                            fg="#7f8c8d",
                                            bg="white")
            self.initial_message.pack(expand=True)
            self.status_bar.config(text=f"Loaded: {os.path.basename(file_path)}")


    def display_uploaded_image(self, image_path):
        try:
            img = Image.open(image_path)
            img.thumbnail((250, 250))
            # Keep a reference to the PhotoImage to prevent garbage collection
            self.upload_display_img = ImageTk.PhotoImage(img)
            self.upload_canvas.delete("all")
            canvas_width = self.upload_canvas.winfo_width()
            canvas_height = self.upload_canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                x = canvas_width // 2
                y = canvas_height // 2
                self.upload_canvas.create_image(x, y, image=self.upload_display_img)
            else:
                self.upload_canvas.config(width=self.upload_display_img.width(),
                                          height=self.upload_display_img.height())
                self.upload_canvas.create_image(0, 0, image=self.upload_display_img, anchor=tk.NW)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")


    def predict_uploaded_image(self):
        """
        Predict the currently loaded image. This function is safe to call
        multiple times; the user can load another image and call it again
        without restarting the program.
        """
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please load an image first!")
            return

        if self.recognizer.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return

        try:
            self.status_bar.config(text="Predicting uploaded image...")
            self.root.update()

            # perform prediction
            result = self.recognizer.predict(self.current_image_path)

            if result:
                self.upload_result = result
                # Update UI with result
                self.display_results(result, source="upload")
                self.status_bar.config(text=f"Upload prediction: {result['arabic_digit']} ({result['english_name']}) - {result['confidence']:.1f}%")
            else:
                messagebox.showerror("Error", "Failed to process image!")
                self.status_bar.config(text="Prediction failed: could not process image.")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.status_bar.config(text="Prediction failed!")


    def predict_drawing(self):
        if self.recognizer.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return

        try:
            self.status_bar.config(text="Predicting drawing...")
            self.root.update()

            img_array = self.drawing_canvas_widget.get_image_array()
            result = self.recognizer.predict_from_array(img_array)

            if result:
                self.draw_result = result
                self.display_results(result, source="draw")
                self.status_bar.config(text=f"Drawing prediction: {result['arabic_digit']} ({result['english_name']}) - {result['confidence']:.1f}%")
            else:
                messagebox.showerror("Error", "Failed to process drawing!")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.status_bar.config(text="Prediction failed!")


    def display_results(self, result, source="upload"):
        for widget in self.result_display.winfo_children():
            widget.destroy()

        source_text = "Uploaded Image" if source == "upload" else "Hand-drawn Digit"
        source_color = self.secondary_color if source == "upload" else self.draw_color

        tk.Label(self.result_display,
                 text=f"Source: {source_text}",
                 font=("Arial", 10, "bold"),
                 bg="white",
                 fg=source_color).pack(anchor=tk.W, pady=(0, 10))

        main_prediction = tk.Frame(self.result_display, bg="white")
        main_prediction.pack(fill=tk.X, pady=(0, 20))

        digit_frame = tk.Frame(main_prediction, bg="white")
        digit_frame.pack(side=tk.LEFT, padx=(0, 20))

        tk.Label(digit_frame,
                 text="Predicted Digit:",
                 font=("Arial", 11),
                 bg="white").pack(anchor=tk.W)

        tk.Label(digit_frame,
                 text=f"{result['arabic_digit']}",
                 font=("Arial", 72, "bold"),
                 bg="white",
                 fg=self.accent_color).pack()

        info_frame = tk.Frame(main_prediction, bg="white")
        info_frame.pack(side=tk.LEFT)

        tk.Label(info_frame,
                 text=f"Numeric: {result['prediction']}",
                 font=("Arial", 16, "bold"),
                 bg="white").pack(anchor=tk.W, pady=(10, 5))

        tk.Label(info_frame,
                 text=f"English: {result['english_name']}",
                 font=("Arial", 14),
                 bg="white").pack(anchor=tk.W, pady=5)

        conf_frame = tk.Frame(self.result_display, bg="white")
        conf_frame.pack(fill=tk.X, pady=(0, 20))

        conf_text = f"Confidence: {result['confidence']:.1f}%"
        tk.Label(conf_frame,
                 text=conf_text,
                 font=("Arial", 14, "bold"),
                 bg="white",
                 fg="#27ae60" if result["confidence"] > 80 else "#e74c3c").pack(anchor=tk.W)

        progress = ttk.Progressbar(conf_frame,
                                   length=250,
                                   mode="determinate",
                                   maximum=100)
        progress["value"] = result["confidence"]
        progress.pack(anchor=tk.W, pady=5)

        if result["confidence"] > 90:
            rating = "Excellent"
            rating_color = "#27ae60"
        elif result["confidence"] > 70:
            rating = "Good"
            rating_color = "#f39c12"
        elif result["confidence"] > 50:
            rating = "Fair"
            rating_color = "#e67e22"
        else:
            rating = "Low"
            rating_color = "#e74c3c"

        tk.Label(conf_frame,
                 text=f"Confidence Rating: {rating}",
                 font=("Arial", 10),
                 bg="white",
                 fg=rating_color).pack(anchor=tk.W)

        prob_frame = tk.LabelFrame(self.result_display,
                                   text="All Probabilities",
                                   font=("Arial", 11, "bold"),
                                   bg="white",
                                   relief=tk.GROOVE)
        prob_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        probs = np.asarray(result["probabilities"])
        for i in range(0, 10, 5):
            row_frame = tk.Frame(prob_frame, bg="white")
            row_frame.pack(fill=tk.X, pady=3)
            for j in range(5):
                digit_idx = i + j
                if digit_idx < 10:
                    prob = float(probs[digit_idx] * 100.0)
                    is_predicted = digit_idx == int(result["prediction"])

                    digit_frame = tk.Frame(row_frame, bg="#f8f9fa" if is_predicted else "white",
                                           relief=tk.RAISED, borderwidth=1)
                    digit_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

                    digit_label = tk.Frame(digit_frame, bg="#f8f9fa" if is_predicted else "white")
                    digit_label.pack(fill=tk.X, padx=5, pady=3)

                    if is_predicted:
                        tk.Label(digit_label, text="⭐", bg="#f8f9fa", font=("Arial", 10)).pack(side=tk.LEFT)

                    tk.Label(digit_label,
                             text=f"{self.recognizer.arabic_digits[digit_idx]}",
                             font=("Arial", 16, "bold"),
                             bg="#f8f9fa" if is_predicted else "white").pack(side=tk.LEFT, padx=5)

                    tk.Label(digit_label,
                             text=f"({digit_idx})",
                             font=("Arial", 10),
                             bg="#f8f9fa" if is_predicted else "white").pack(side=tk.LEFT)

                    prob_bar_frame = tk.Frame(digit_frame, bg="#f8f9fa" if is_predicted else "white")
                    prob_bar_frame.pack(fill=tk.X, padx=5, pady=(0, 3))

                    small_progress = ttk.Progressbar(prob_bar_frame,
                                                     length=100,
                                                     mode="determinate",
                                                     maximum=100)
                    small_progress["value"] = prob
                    small_progress.pack(side=tk.LEFT, padx=(0, 5))

                    tk.Label(prob_bar_frame,
                             text=f"{prob:.1f}%",
                             font=("Arial", 8),
                             bg="#f8f9fa" if is_predicted else "white").pack(side=tk.LEFT)


    def compare_results(self):
        if not self.upload_result and not self.draw_result:
            messagebox.showinfo("No Results", "Please predict both an uploaded image and a drawing first!")
            return

        compare_window = tk.Toplevel(self.root)
        compare_window.title("Comparison Results")
        compare_window.geometry("600x400")
        compare_window.configure(bg=self.bg_color)

        tk.Label(compare_window,
                 text="🔍 Comparison Results",
                 font=("Arial", 16, "bold"),
                 bg=self.bg_color).pack(pady=20)

        table_frame = tk.Frame(compare_window, bg="white")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        headers = ["Method", "Digit", "Arabic", "English", "Confidence"]
        for col, header in enumerate(headers):
            tk.Label(table_frame,
                     text=header,
                     font=("Arial", 11, "bold"),
                     bg="#34495e",
                     fg="white",
                     width=15).grid(row=0, column=col, padx=1, pady=1, sticky="nsew")

        row_num = 1
        if self.upload_result:
            row_data = [
                "Uploaded",
                str(self.upload_result["prediction"]),
                self.upload_result["arabic_digit"],
                self.upload_result["english_name"],
                f"{self.upload_result['confidence']:.1f}%"
            ]
            for col, data in enumerate(row_data):
                tk.Label(table_frame,
                         text=data,
                         font=("Arial", 10),
                         bg="#ecf0f1",
                         width=15).grid(row=row_num, column=col, padx=1, pady=1, sticky="nsew")
            row_num += 1

        if self.draw_result:
            row_data = [
                "Hand-drawn",
                str(self.draw_result["prediction"]),
                self.draw_result["arabic_digit"],
                self.draw_result["english_name"],
                f"{self.draw_result['confidence']:.1f}%"
            ]
            for col, data in enumerate(row_data):
                tk.Label(table_frame,
                         text=data,
                         font=("Arial", 10),
                         bg="#d5f4e6",
                         width=15).grid(row=row_num, column=col, padx=1, pady=1, sticky="nsew")
            row_num += 1

        for i in range(len(headers)):
            table_frame.grid_columnconfigure(i, weight=1)

        tk.Button(compare_window,
                  text="Close",
                  command=compare_window.destroy,
                  bg=self.accent_color,
                  fg="white",
                  font=("Arial", 11),
                  width=20).pack(pady=20)


def print_result_cli(result, recognizer):
    print("\n" + "=" * 60)
    print("🎯 PREDICTION RESULT")
    print("=" * 60)
    print(f"\n🔢 Digit: {result['prediction']}")
    print(f"   Arabic: {result['arabic_digit']}")
    print(f"   English: {result['english_name']}")
    print(f"   Confidence: {result['confidence']:.1f}%")
    print(f"\n📊 Probabilities:")
    probs = np.asarray(result["probabilities"])
    for i, prob in enumerate(probs):
        star = "⭐" if i == result["prediction"] else "  "
        print(f"   {star} {recognizer.arabic_digits[i]} ({i}): {prob * 100:6.2f}%")


def main_cli():
    print("=" * 60)
    print("🔢 Arabic Digit Recognition - Test Interface")
    print("=" * 60)

    recognizer = DigitRecognizer()

    if recognizer.model is None:
        print("Please train the model first!")
        return

    while True:
        print("\nOptions:")
        print("1. Test with image path")
        print("2. Test multiple images in folder")
        print("3. Show model info")
        print("4. Open GUI (Draw & Predict)")
        print("5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            image_path = input("Enter image path: ").strip()
            if not os.path.exists(image_path):
                print("❌ Image not found!")
                continue
            result = recognizer.predict(image_path)
            if result:
                print_result_cli(result, recognizer)

        elif choice == "2":
            folder_path = input("Enter folder path containing images: ").strip()
            if not os.path.exists(folder_path):
                print("❌ Folder not found!")
                continue

            image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".gif")
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

            if not images:
                print("❌ No images found in folder!")
                continue

            print(f"\nFound {len(images)} images")
            for img_file in images:
                img_path = os.path.join(folder_path, img_file)
                print(f"\nTesting: {img_file}")
                result = recognizer.predict(img_path)
                if result:
                    print_result_cli(result, recognizer)

        elif choice == "3":
            if recognizer.model_info:
                print("\n📋 MODEL INFORMATION")
                print(f"   Algorithm: {recognizer.model_info.get('algorithm', 'Unknown')}")
                print(f"   Test Accuracy: {recognizer.model_info.get('test_accuracy', 0.0):.4f}")
                if "train_accuracy" in recognizer.model_info:
                    print(f"   Train Accuracy: {recognizer.model_info['train_accuracy']:.4f}")
                if "training_time" in recognizer.model_info:
                    print(f"   Training Time: {recognizer.model_info['training_time']:.2f}s")
            else:
                print("No model information available")

        elif choice == "4":
            print("\nOpening GUI interface...")
            main_gui()
            return

        elif choice == "5":
            print("\nGoodbye!")
            break

        else:
            print("Invalid choice!")


def main_gui():
    root = tk.Tk()
    app = DigitRecognizerGUI(root)

    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    print("=" * 60)
    print("Arabic Digit Recognition System")
    print("=" * 60)
    print("\nSelect Interface:")
    print("1. Command Line Interface (CLI)")
    print("2. Graphical User Interface (GUI) with Drawing")

    choice = input("\nEnter choice (1 or 2): ").strip()
    if choice == "1":
        main_cli()
    else:
        main_gui()