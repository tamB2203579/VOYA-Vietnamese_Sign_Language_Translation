import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
import os
import cv2
import mediapipe as mp
import json
import albumentations as A
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Utils ---

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- Capture Images Thread ---
class CaptureImagesThread(threading.Thread):
    def __init__(self, label, save_dir, interval, max_images, update_callback):
        super().__init__()
        self.label = label
        self.save_dir = save_dir
        self.interval = interval
        self.max_images = max_images
        self.update_callback = update_callback
        self._stop_flag = False
        self._pause_flag = False  # Thêm dòng này

    def stop(self):
        self._stop_flag = True

    def pause(self):
        self._pause_flag = True

    def resume(self):
        self._pause_flag = False

    def run(self):
        save_path = os.path.join(self.save_dir, self.label)
        ensure_dir(save_path)

        cap = cv2.VideoCapture(0)
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

        # Đếm ngược 3 2 1 trước khi chụp
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                self.update_callback("Failed to grab frame")
                break
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Starting in {i}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
            cv2.imshow(f"Capture - Label: {self.label}", display_frame)
            cv2.waitKey(1000)  # 1 giây
        cv2.destroyAllWindows()

        count = 0
        last_time = 0

        while not self._stop_flag and count < self.max_images:
            if self._pause_flag:
                time.sleep(0.1)
                continue
            ret, frame = cap.read()
            if not ret:
                self.update_callback("Failed to grab frame")
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            display_frame = frame.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                    )

            cv2.putText(display_frame, f"Captured: {count}/{self.max_images}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow(f"Capture - Label: {self.label} (Press q to stop)", display_frame)

            current_time = time.time()
            if current_time - last_time >= self.interval and results.multi_hand_landmarks:
                img_name = os.path.join(save_path, f"{self.label}_{count:04d}.jpg")
                cv2.imwrite(img_name, frame)
                count += 1
                self.update_callback(f"Saved {img_name}")
                last_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.update_callback("User stopped capture.")
                break

        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        self.update_callback("Capture thread ended.")

# --- Gallery Viewer Thread ---
class GalleryViewerThread(threading.Thread):
    def __init__(self, folder, update_callback):
        super().__init__()
        self.folder = folder
        self.update_callback = update_callback
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        images = sorted([f for f in os.listdir(self.folder) if f.lower().endswith(('.jpg', '.png'))])
        if not images:
            self.update_callback(f"No images found in {self.folder}")
            return

        index = 0
        while not self._stop_flag:
            img_path = os.path.join(self.folder, images[index])
            img = cv2.imread(img_path)
            if img is None:
                self.update_callback(f"Failed to load {img_path}")
                index = (index + 1) % len(images)
                continue

            cv2.imshow(f"Gallery: {self.folder} - Press ESC to quit, A/D to navigate", img)
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break
            elif key == ord('d'):
                index = (index + 1) % len(images)
            elif key == ord('a'):
                index = (index - 1) % len(images)

        cv2.destroyAllWindows()
        self.update_callback("Gallery viewer ended.")

# --- Augmentation function ---
def augment_images(input_dir, output_dir, augmentations_per_image=5, callback=None):
    ensure_dir(output_dir)

    transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.GaussNoise(p=0.15),
        A.ISONoise(p=0.1),
        A.MotionBlur(p=0.1),
        A.MedianBlur(blur_limit=3, p=0.05),
        A.Blur(blur_limit=3, p=0.05),
        A.HueSaturationValue(p=0.2),
        # Các phép biến đổi mạnh để xác suất thấp hoặc loại bỏ:
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.03, rotate_limit=10, p=0.1),
        # Nếu muốn giữ, giảm p xuống thấp:
        A.OneOf([
            A.OpticalDistortion(p=0.05),
            A.GridDistortion(p=0.05),
            A.PiecewiseAffine(p=0.05),
        ], p=0.05),
    ])

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            if callback:
                callback(f"Failed to load {img_path}")
            continue

        base_name = os.path.splitext(img_name)[0]
        for i in range(augmentations_per_image):
            augmented = transform(image=image)
            aug_img = augmented['image']
            save_path = os.path.join(output_dir, f"{base_name}_aug_{i}.jpg")
            cv2.imwrite(save_path, aug_img)
            if callback:
                callback(f"Saved augmented {save_path}")

    if callback:
        callback("Augmentation completed.")

# --- Extract Keypoints function ---
def extract_keypoints_from_folder(input_dir, output_dir, callback=None):
    ensure_dir(output_dir)

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            if callback:
                callback(f"Failed to read {img_path}")
            continue

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            if callback:
                callback(f"No hand detected in {img_name}, skipping")
            continue

        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.append({'x': lm.x, 'y': lm.y, 'z': lm.z})

        json_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.json')
        with open(json_path, 'w') as f:
            json.dump(keypoints, f)

        if callback:
            callback(f"Saved keypoints to {json_path}")

    hands.close()
    if callback:
        callback("Keypoints extraction done.")

# --- Record Dynamic Sequence Thread ---
class RecordSequenceThread(threading.Thread):
    def __init__(self, label, seq_num, save_dir, duration, fps, update_callback):
        super().__init__()
        self.label = label
        self.seq_num = seq_num
        self.save_dir = save_dir
        self.duration = duration
        self.fps = fps
        self.update_callback = update_callback
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        sequence_path = os.path.join(self.save_dir, self.label, f"sequence_{self.seq_num:03d}")
        ensure_dir(sequence_path)

        cap = cv2.VideoCapture(0)
        frame_time = 1.0 / self.fps
        frame_count = int(self.duration * self.fps)

        # Đếm ngược trước khi ghi
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                self.update_callback("Failed to grab frame")
                break
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Recording in {i}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
            cv2.imshow(f"Recording {self.label} Seq {self.seq_num}", display_frame)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()

        for i in range(frame_count):
            if self._stop_flag:
                self.update_callback("User stopped recording.")
                break

            ret, frame = cap.read()
            if not ret:
                self.update_callback("Failed to grab frame")
                break

            cv2.putText(frame, f"{i+1}/{frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow(f"Recording {self.label} Seq {self.seq_num}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.update_callback("User stopped recording (q pressed).")
                break

            filename = os.path.join(sequence_path, f"{self.label}_{i:04d}.jpg")
            cv2.imwrite(filename, frame)

            time.sleep(frame_time)

        cap.release()
        cv2.destroyAllWindows()
        self.update_callback(f"Recording ended. Saved {frame_count} frames to {sequence_path}")

# --- Extract Keypoints from Sequences function ---
def extract_keypoints_from_sequences(input_dir, output_dir, callback=None):
    ensure_dir(output_dir)

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

    sequences = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for seq in sequences:
        seq_input_path = os.path.join(input_dir, seq)
        seq_output_path = os.path.join(output_dir, seq)
        ensure_dir(seq_output_path)

        images = sorted([f for f in os.listdir(seq_input_path) if f.lower().endswith(('.jpg', '.png'))])
        for img_name in images:
            img_path = os.path.join(seq_input_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                if callback:
                    callback(f"Failed to load {img_path}")
                continue

            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                if callback:
                    callback(f"No hand detected in {img_name}, skipping")
                continue

            hand_landmarks = results.multi_hand_landmarks[0]
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append({'x': lm.x, 'y': lm.y, 'z': lm.z})

            json_path = os.path.join(seq_output_path, os.path.splitext(img_name)[0] + '.json')
            with open(json_path, 'w') as f:
                json.dump(keypoints, f)

            if callback:
                callback(f"Saved keypoints to {json_path}")

        if callback:
            callback(f"Finished sequence {seq}")

    hands.close()
    if callback:
        callback("Extraction from sequences done.")

# --- Main GUI App ---

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Data Collection & Preprocessing")

        self.save_dir = "data/images"
        self.seq_save_dir = "data/sequences"
        self.aug_output_dir = "data/augmented"
        self.keypoints_dir = "data/keypoints"
        self.seq_keypoints_dir = "data/sequence_keypoints"

        self.capture_thread = None
        self.gallery_thread = None
        self.record_seq_thread = None

        self.create_widgets()

    def create_widgets(self):
        # Label Input
        label_frame = ttk.LabelFrame(self.root, text="Label & Parameters")
        label_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(label_frame, text="Label (e.g. A, B, C):").grid(row=0, column=0, sticky="w")
        self.label_entry = ttk.Entry(label_frame, width=10)
        self.label_entry.grid(row=0, column=1, sticky="w")

        ttk.Label(label_frame, text="Interval (sec):").grid(row=1, column=0, sticky="w")
        self.interval_entry = ttk.Entry(label_frame, width=10)
        self.interval_entry.insert(0, "1")
        self.interval_entry.grid(row=1, column=1, sticky="w")

        ttk.Label(label_frame, text="Max Images:").grid(row=2, column=0, sticky="w")
        self.max_images_entry = ttk.Entry(label_frame, width=10)
        self.max_images_entry.insert(0, "50")
        self.max_images_entry.grid(row=2, column=1, sticky="w")

        # Buttons Frame
        buttons_frame = ttk.LabelFrame(self.root, text="Capture & Gallery")
        buttons_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.btn_start_capture = ttk.Button(buttons_frame, text="Start Capture", command=self.start_capture)
        self.btn_start_capture.grid(row=0, column=0, padx=5, pady=5)

        self.btn_stop_capture = ttk.Button(buttons_frame, text="Stop Capture", command=self.stop_capture)
        self.btn_stop_capture.grid(row=0, column=1, padx=5, pady=5)

        self.btn_view_gallery = ttk.Button(buttons_frame, text="View Gallery", command=self.view_gallery)
        self.btn_view_gallery.grid(row=0, column=2, padx=5, pady=5)

        self.btn_pause_capture = ttk.Button(buttons_frame, text="Pause Capture", command=self.pause_capture)
        self.btn_pause_capture.grid(row=0, column=3, padx=5, pady=5)
        self.btn_resume_capture = ttk.Button(buttons_frame, text="Resume Capture", command=self.resume_capture)
        self.btn_resume_capture.grid(row=0, column=4, padx=5, pady=5)

        # Augmentation Frame
        aug_frame = ttk.LabelFrame(self.root, text="Augmentation")
        aug_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        self.btn_augment = ttk.Button(aug_frame, text="Run Augmentation", command=self.run_augmentation)
        self.btn_augment.grid(row=0, column=0, padx=5, pady=5)

        # Keypoints Extraction Frame
        keypoints_frame = ttk.LabelFrame(self.root, text="Keypoints Extraction")
        keypoints_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self.btn_extract_keypoints = ttk.Button(keypoints_frame, text="Extract Keypoints", command=self.extract_keypoints)
        self.btn_extract_keypoints.grid(row=0, column=0, padx=5, pady=5)

        self.btn_extract_aug_keypoints = ttk.Button(keypoints_frame, text="Extract Keypoints from Augmented", command=self.extract_keypoints_from_augmented)
        self.btn_extract_aug_keypoints.grid(row=1, column=0, padx=5, pady=5)

        # Dynamic Sequence Recording Frame
        seq_frame = ttk.LabelFrame(self.root, text="Dynamic Sequence Recording")
        seq_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(seq_frame, text="Label:").grid(row=0, column=0)
        self.seq_label_entry = ttk.Entry(seq_frame, width=10)
        self.seq_label_entry.grid(row=0, column=1)

        ttk.Label(seq_frame, text="Sequence #:").grid(row=0, column=2)
        self.seq_num_entry = ttk.Entry(seq_frame, width=5)
        self.seq_num_entry.grid(row=0, column=3)

        ttk.Label(seq_frame, text="Duration (s):").grid(row=1, column=0)
        self.seq_duration_entry = ttk.Entry(seq_frame, width=10)
        self.seq_duration_entry.insert(0, "5")
        self.seq_duration_entry.grid(row=1, column=1)

        ttk.Label(seq_frame, text="FPS:").grid(row=1, column=2)
        self.seq_fps_entry = ttk.Entry(seq_frame, width=5)
        self.seq_fps_entry.insert(0, "10")
        self.seq_fps_entry.grid(row=1, column=3)

        self.btn_start_seq = ttk.Button(seq_frame, text="Start Recording Sequence", command=self.start_sequence_recording)
        self.btn_start_seq.grid(row=2, column=0, columnspan=2, pady=5)

        self.btn_stop_seq = ttk.Button(seq_frame, text="Stop Recording", command=self.stop_sequence_recording)
        self.btn_stop_seq.grid(row=2, column=2, columnspan=2, pady=5)

        # Extract keypoints from sequences
        self.btn_extract_seq_keypoints = ttk.Button(seq_frame, text="Extract Keypoints from Sequences", command=self.extract_seq_keypoints)
        self.btn_extract_seq_keypoints.grid(row=3, column=0, columnspan=4, pady=5)

        # Status Frame
        status_frame = ttk.LabelFrame(self.root, text="Status / Logs")
        status_frame.grid(row=5, column=0, padx=10, pady=10, sticky="ew")

        self.status_text = tk.Text(status_frame, height=10, width=60)
        self.status_text.grid(row=0, column=0)

        # Scrollbar for status
        scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.status_text['yscrollcommand'] = scrollbar.set

    def log(self, message):
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)

    def start_capture(self):
        label = self.label_entry.get().strip()
        if not label:
            messagebox.showerror("Error", "Please enter a label.")
            return
        try:
            interval = float(self.interval_entry.get())
            max_images = int(self.max_images_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid interval or max images value.")
            return

        if self.capture_thread and self.capture_thread.is_alive():
            messagebox.showwarning("Warning", "Capture is already running.")
            return

        self.capture_thread = CaptureImagesThread(label, self.save_dir, interval, max_images, self.log)
        self.capture_thread.start()
        self.log(f"Started capturing images for label '{label}'")

    def stop_capture(self):
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.stop()
            self.capture_thread.join()
            self.log("Stopped capturing images.")
        else:
            self.log("No capture running.")

    def view_gallery(self):
        label = self.label_entry.get().strip()
        if not label:
            messagebox.showerror("Error", "Please enter a label.")
            return
        folder = os.path.join(self.save_dir, label)
        if not os.path.exists(folder):
            messagebox.showerror("Error", f"No images found for label '{label}'")
            return
        # Open new window to display images
        gallery_win = tk.Toplevel(self.root)
        gallery_win.title(f"Gallery for label '{label}'")

        images = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))])
        if not images:
            messagebox.showinfo("Info", "No images to display.")
            return

        canvas = tk.Canvas(gallery_win)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(gallery_win, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor='nw')

        # Load and display images as thumbnails
        thumbs = []
        for img_file in images:
            img_path = os.path.join(folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (120, 90))
            pil_img = Image.fromarray(img)  # Convert numpy array to PIL image
            photo = ImageTk.PhotoImage(pil_img)  # Convert PIL image to Tkinter image
            label_img = ttk.Label(frame, image=photo)
            label_img.image = photo  # Prevent garbage collection
            label_img.pack(padx=5, pady=5)
            thumbs.append(label_img)


    def run_augmentation(self):
        label = self.label_entry.get().strip()
        if not label:
            messagebox.showerror("Error", "Please enter a label.")
            return
        input_dir = os.path.join(self.save_dir, label)
        output_dir = os.path.join(self.aug_output_dir, label)
        if not os.path.exists(input_dir):
            messagebox.showerror("Error", f"No images found for label '{label}'")
            return
        images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not images:
            messagebox.showerror("Error", f"No images to augment for label '{label}'")
            return
        self.log(f"Augmentation started for label '{label}'...")
        threading.Thread(
            target=augment_images,
            args=(input_dir, output_dir, 5, self.log),
            daemon=True
        ).start()

    def extract_keypoints(self):
        label = self.label_entry.get().strip()
        if not label:
            messagebox.showerror("Error", "Please enter a label.")
            return
        input_dir = os.path.join(self.save_dir, label)
        output_dir = os.path.join(self.keypoints_dir, label)
        if not os.path.exists(input_dir):
            messagebox.showerror("Error", f"No images found for label '{label}'")
            return
        images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not images:
            messagebox.showerror("Error", f"No images to extract keypoints for label '{label}'")
            return
        self.log(f"Extracting keypoints from images for label '{label}'...")
        threading.Thread(
            target=extract_keypoints_from_folder,
            args=(input_dir, output_dir, self.log),
            daemon=True
        ).start()

    def start_sequence_recording(self):
        label = self.seq_label_entry.get().strip()
        seq_num = self.seq_num_entry.get().strip()
        try:
            duration = float(self.seq_duration_entry.get())
            fps = int(self.seq_fps_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid duration or FPS value.")
            return
        if not label or not seq_num:
            messagebox.showerror("Error", "Please enter label and sequence number.")
            return

        if self.record_seq_thread and self.record_seq_thread.is_alive():
            messagebox.showwarning("Warning", "Sequence recording is already running.")
            return

        self.record_seq_thread = RecordSequenceThread(label, seq_num, self.seq_save_dir, duration, fps, self.log)
        self.record_seq_thread.start()
        self.log(f"Started recording sequence {seq_num} for label '{label}'")

    def stop_sequence_recording(self):
        if self.record_seq_thread and self.record_seq_thread.is_alive():
            self.record_seq_thread.stop()
            self.record_seq_thread.join()
            self.log("Stopped sequence recording.")
        else:
            self.log("No sequence recording running.")

    def extract_seq_keypoints(self):
        self.log("Extracting keypoints from sequences...")
        threading.Thread(target=extract_keypoints_from_sequences, args=(self.seq_save_dir, self.seq_keypoints_dir, self.log), daemon=True).start()

    def extract_keypoints_from_augmented(self):
        label = self.label_entry.get().strip()
        if not label:
            messagebox.showerror("Error", "Please enter a label.")
            return
        input_dir = os.path.join(self.aug_output_dir, label)
        output_dir = os.path.join(self.keypoints_dir, f"{label}_aug")
        if not os.path.exists(input_dir):
            messagebox.showerror("Error", f"No augmented images found for label '{label}'")
            return
        images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not images:
            messagebox.showerror("Error", f"No augmented images to extract keypoints for label '{label}'")
            return
        self.log(f"Extracting keypoints from augmented images for label '{label}'...")
        threading.Thread(
            target=extract_keypoints_from_folder,
            args=(input_dir, output_dir, self.log),
            daemon=True
        ).start()

    def pause_capture(self):
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.pause()
            self.log("Capture paused.")
        else:
            self.log("No capture running.")

    def resume_capture(self):
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.resume()
            self.log("Capture resumed.")
        else:
            self.log("No capture running.")


if __name__ == "__main__":
    import tkinter as tk
    from tkinter import ttk, messagebox
    import threading
    import cv2
    import mediapipe as mp
    import time
    import os
    import json

    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
