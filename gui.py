import cv2
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import torch
import torchvision.transforms as transforms
from transformers import ViTModel, ViTConfig, logging
import os
from reporting import generate_report
from database import DatabaseManager

# Suppress unnecessary warnings
logging.set_verbosity_error()

# Constants
IMG_SIZE = 224
CLASS_MAPPING = {
    0: "Arson",
    1: "Fighting",
    2: "RoadAccidents",
    3: "Robbery",
    4: "Normal"
}
CONFIDENCE_THRESHOLD = 0.75

# Define the VideoViT model
class VideoViT(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            config=config,
            add_pooling_layer=False
        )
        self.temporal_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=768, nhead=8),
            num_layers=2
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, num_classes)
        )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        vit_outputs = self.vit(x).last_hidden_state
        cls_tokens = vit_outputs[:, 0, :].view(batch_size, timesteps, -1)
        temporal_features = self.temporal_encoder(cls_tokens)
        pooled = temporal_features.mean(dim=1)
        return self.classifier(pooled)


class SmartMonitoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Monitoring & Anomaly Detection")
        self.root.state('zoomed')

        # Theme state
        self.current_theme = "dark"

        # Background image paths
        self.bg_image_path = "assets/111.jpg"
        self.bg_image_light_path = "assets/222.png"  

        # Report directory and database setup
        self.report_dir = "camera_reports"
        os.makedirs(self.report_dir, exist_ok=True)
        self.db_path = os.path.join(self.report_dir, "monitoring.db")
        self.db_manager = DatabaseManager(self.db_path)
        self.users = self.db_manager.get_users()

        # OpenAI API key (replace with your actual key)
        self.api_key = "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"

        # Log file for login records
        self.log_file_path = "monitoring.db"

        # Initialize UI
        self.create_widgets()
        self.apply_theme()

        # Video camera setup
        self.num_cameras = 8
        self.captures = [cv2.VideoCapture(f"videos/{i+1}.mp4") for i in range(self.num_cameras)]
        self.camera_labels = []

        # Model setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VideoViT()
        try:
            self.model.load_state_dict(torch.load("best_model.pth", map_location=self.device))
        except FileNotFoundError:
            print("Warning: best_model.pth not found.")
        self.model.to(self.device)
        self.model.eval()
        self.model_lock = threading.Lock()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Stop events for video threads
        self.stop_events = [threading.Event() for _ in range(self.num_cameras)]

        # Back button
        self.back_button = ttk.Button(self.root, text="Go Back", command=self.go_back)

    def create_widgets(self):
        # Background image label
        self.bg_label = tk.Label(self.root)

        # Styling
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Main frames
        self.main_frame = ttk.Frame(self.root)
        self.report_frame = ttk.Frame(self.root)
        self.admin_frame = ttk.Frame(self.root)

        # Create login frame initially
        self.create_login_frame()

        # Enhanced Report Treeview
        self.report_tree = ttk.Treeview(
            self.report_frame,
            columns=("Timestamp", "Event", "Cam Num", "Confidence"),
            show="headings",
            height=20
        )
        self.report_tree.heading("Timestamp", text="Timestamp")
        self.report_tree.heading("Event", text="Event")
        self.report_tree.heading("Cam Num", text="Cam Num")
        self.report_tree.heading("Confidence", text="Confidence")
        self.report_tree.column("Timestamp", width=150)
        self.report_tree.column("Event", width=200)
        self.report_tree.column("Cam Num", width=80)
        self.report_tree.column("Confidence", width=100)
        self.report_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.report_tree.bind("<<TreeviewSelect>>", self.show_anomaly_details)
        scroll = ttk.Scrollbar(self.report_frame, orient="vertical", command=self.report_tree.yview)
        scroll.pack(side="right", fill="y")
        self.report_tree.configure(yscrollcommand=scroll.set)

        # Admin UI
        self.operator_listbox = tk.Listbox(self.admin_frame, font=("Arial", 14))
        self.operator_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        btn_frame = ttk.Frame(self.admin_frame)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Add Operator", command=self.add_operator).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Delete Operator", command=self.delete_operator).grid(row=0, column=1, padx=5)

        # Mode toggle button
        self.mode_button = ttk.Button(self.root, text="Light Mode", command=self.toggle_theme)
        self.mode_button.place(relx=0.80, rely=0.97, anchor="se")

    def create_login_frame(self):
        self.login_frame = ttk.Frame(self.root, padding=20)
        self.login_frame.place(relx=0.5, rely=0.5, anchor="center")
        ttk.Label(self.login_frame, text="Email:", font=("Arial", 14)).grid(row=0, column=0, padx=10, pady=15, sticky="w")
        self.email_entry = ttk.Entry(self.login_frame, font=("Arial", 14))
        self.email_entry.grid(row=0, column=1, pady=5)
        ttk.Label(self.login_frame, text="Password:", font=("Arial", 14)).grid(row=1, column=0, padx=10, pady=15, sticky="w")
        self.password_entry = ttk.Entry(self.login_frame, show="*", font=("Arial", 14))
        self.password_entry.grid(row=1, column=1, pady=5)
        ttk.Button(self.login_frame, text="Login", command=self.login).grid(row=2, column=0, columnspan=2, pady=10)

    def apply_theme(self):
        if self.current_theme == "dark":
            bg_color = "black"
            fg_color = "white"
            btn_bg = "#444"
            btn_active = "#555"
            tree_bg = "black"
            tree_fg = "white"
            tree_sel = "#555"

            try:
                self.bg_image = Image.open(self.bg_image_path).resize(
                    (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
                )
                self.bg_photo = ImageTk.PhotoImage(self.bg_image)
                self.bg_label.config(image=self.bg_photo)
                self.bg_label.place(relwidth=1, relheight=1)
            except Exception as e:
                print(f"Error loading dark mode background: {e}")
                self.bg_label.place_forget()
        else:
            bg_color = "white"
            fg_color = "black"
            btn_bg = "#f0f0f0"
            btn_active = "#e0e0e0"
            tree_bg = "white"
            tree_fg = "black"
            tree_sel = "#d9d9d9"

            try:
                self.bg_image_light = Image.open(self.bg_image_light_path).resize(
                    (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
                )
                self.bg_photo_light = ImageTk.PhotoImage(self.bg_image_light)
                self.bg_label.config(image=self.bg_photo_light)
                self.bg_label.place(relwidth=1, relheight=1)
            except Exception as e:
                print(f"Error loading light mode background: {e}")
                self.bg_label.place_forget()

        # Apply styles
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabel", background=bg_color, foreground=fg_color)
        self.style.configure("TEntry", fieldbackground=bg_color, foreground=fg_color)
        self.style.configure("TButton",
                            background=btn_bg,
                            foreground=fg_color,
                            bordercolor=btn_bg,
                            lightcolor=btn_bg,
                            darkcolor=btn_bg)
        self.style.map("TButton", background=[('active', btn_active)])
        self.style.configure("Treeview",
                            background=tree_bg,
                            foreground=tree_fg,
                            fieldbackground=tree_bg,
                            rowheight=30)
        self.style.configure("Treeview.Heading",
                            background=tree_bg,
                            foreground=fg_color,
                            relief="flat")
        self.style.map("Treeview", background=[('selected', tree_sel)])
        self.root.configure(bg=bg_color)

        for widget in [self.operator_listbox, self.report_tree]:
            if isinstance(widget, tk.Listbox):
                widget.config(bg=bg_color, fg=fg_color)

    def toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.mode_button.config(text="Light Mode" if self.current_theme == "dark" else "Dark Mode")
        self.apply_theme()

    def login(self):
        email = self.email_entry.get().strip()
        pwd = self.password_entry.get().strip()
        if hasattr(self, 'login_warning_label'):
            self.login_warning_label.destroy()
        user = self.users.get(email)
        if user and user['password'] == pwd:
            with open(self.log_file_path, "a") as log_file:
                log_file.write(f"{email},{pwd}\n")
            self.login_frame.destroy()
            if user['role'] == 'admin':
                self.show_admin_interface()
            else:
                self.show_operator_interface()
        else:
            self.login_warning_label = tk.Label(
                self.login_frame,
                text="Invalid email or password",
                fg="orange", bg="black",
                font=("Arial", 10)
            )
            self.login_warning_label.grid(row=3, column=0, columnspan=2, pady=(5, 0))

    def show_operator_interface(self):
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0, pady=0)
        self.report_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._create_camera_grid()
        self.stop_events = [threading.Event() for _ in range(self.num_cameras)]
        self._start_video_threads()
        self.back_button.place(relx=0.90, rely=0.97, anchor="se")
        self.current_left_frame = self.main_frame

    def show_admin_interface(self):
        self.admin_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0, pady=0)
        self.report_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.operator_listbox.delete(0, tk.END)
        for email, info in self.users.items():
            if info['role'] == 'operator':
                self.operator_listbox.insert(tk.END, f"{info['name']} <{email}>")
        self.back_button.place(relx=0.90, rely=0.97, anchor="se")
        self.current_left_frame = self.admin_frame

    def go_back(self):
        self.current_left_frame.pack_forget()
        self.report_frame.pack_forget()
        self.back_button.place_forget()
        for event in self.stop_events:
            event.set()
        self.create_login_frame()

    def _create_camera_grid(self):
        rows, cols = 4, 2
        for i in range(self.num_cameras):
            frame = ttk.LabelFrame(self.main_frame, text=f"Camera {i+1}")
            frame.grid(row=i // cols, column=i % cols, padx=10, pady=10, sticky="nsew")
            label = tk.Label(frame)
            label.pack(fill=tk.BOTH, expand=True)
            self.camera_labels.append(label)
        for i in range(rows):
            self.main_frame.grid_rowconfigure(i, weight=1)
        for j in range(cols):
            self.main_frame.grid_columnconfigure(j, weight=1)

    def _start_video_threads(self):
        for idx in range(self.num_cameras):
            threading.Thread(target=self._update_camera, args=(idx,), daemon=True).start()

    def _update_camera(self, idx):
        buf = []
        while not self.stop_events[idx].is_set():
            ret, frame = self.captures[idx].read()
            if not ret:
                self.captures[idx].set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (400, 300))
            img = ImageTk.PhotoImage(Image.fromarray(resized))
            self.camera_labels[idx].imgtk = img
            self.camera_labels[idx].config(image=img)
            buf.append(Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE)))
            if len(buf) == 16:
                threading.Thread(target=self._infer, args=(buf, idx), daemon=True).start()
                buf = []
            time.sleep(0.03)

    def _infer(self, frames, idx):
        t = torch.stack([self.transform(f) for f in frames]).unsqueeze(0).to(self.device)
        with self.model_lock, torch.no_grad():
            out = self.model(t)
            probs = torch.nn.functional.softmax(out, dim=1)
            conf, i = torch.max(probs, 1)
        cls = CLASS_MAPPING[i.item()]
        if cls != "Normal" and conf.item() > CONFIDENCE_THRESHOLD:
            self.report_anomaly(idx, cls, conf.item(), frames)

    def report_anomaly(self, idx, cls, conf, frames):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"anomaly_{ts}_cam{idx+1}"
        folder_path = os.path.join(self.report_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        for i, frame in enumerate(frames):
            frame_path = os.path.join(folder_path, f"frame_{i:03d}.jpg")
            frame.save(frame_path)
        with open(os.path.join(folder_path, "predicted_class.txt"), "w") as f:
            f.write(cls)
        summary, frame_data = generate_report(folder_path, self.api_key)
        anomaly_id = self.db_manager.insert_anomaly(ts, cls, f"Cam {idx+1}", conf, summary)
        for path, caption in frame_data:
            self.db_manager.insert_snapshot(anomaly_id, path, caption)
        self.report_tree.insert("", "end", iid=anomaly_id, values=(ts, cls, f"Cam {idx+1}", f"{conf:.2%}"))

    def show_anomaly_details(self, event):
        selected_item = self.report_tree.selection()
        if not selected_item:
            return
        anomaly_id = selected_item[0]
        anomaly = self.db_manager.get_anomaly(anomaly_id)
        snapshots = self.db_manager.get_snapshots(anomaly_id)
        if anomaly:
            details_win = tk.Toplevel(self.root)
            details_win.title("Anomaly Details")
            details_win.geometry("800x600")
            info_text = f"Timestamp: {anomaly[1]}\nEvent: {anomaly[2]}\nCam Num: {anomaly[3]}\nConfidence: {anomaly[4]:.2%}\nReport:\n{anomaly[5]}"
            tk.Label(details_win, text=info_text, justify="left").pack(pady=10)
            for path, caption in snapshots:
                img = Image.open(path)
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)
                lbl = tk.Label(details_win, image=photo)
                lbl.image = photo
                lbl.pack()
                tk.Label(details_win, text=caption).pack()

    def add_operator(self):
        win = tk.Toplevel(self.root)
        win.title("Add Operator")
        win.configure(bg="#2e2e2e")
        win.transient(self.root)
        win.grab_set()
        frm = ttk.Frame(win, padding=20, style="TFrame")
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Email:", style="TLabel").grid(row=0, column=0, sticky="e", pady=5)
        emE = ttk.Entry(frm, font=("Arial", 12))
        emE.grid(row=0, column=1, pady=5)
        email_warn = tk.Label(frm, text="", bg="#2e2e2e", fg="orange", font=("Arial", 10))
        email_warn.grid(row=1, column=1, sticky="w")
        ttk.Label(frm, text="Name:", style="TLabel").grid(row=2, column=0, sticky="e", pady=5)
        nmE = ttk.Entry(frm, font=("Arial", 12))
        nmE.grid(row=2, column=1, pady=5)
        name_warn = tk.Label(frm, text="", bg="#2e2e2e", fg="orange", font=("Arial", 10))
        name_warn.grid(row=3, column=1, sticky="w")
        ttk.Label(frm, text="Password:", style="TLabel").grid(row=4, column=0, sticky="e", pady=5)
        pwE = ttk.Entry(frm, show="*", font=("Arial", 12))
        pwE.grid(row=4, column=1, pady=5)
        pw_warn = tk.Label(frm, text="", bg="#2e2e2e", fg="orange", font=("Arial", 10))
        pw_warn.grid(row=5, column=1, sticky="w")
        ttk.Label(frm, text="Confirm Password:", style="TLabel").grid(row=6, column=0, sticky="e", pady=5)
        cpE = ttk.Entry(frm, show="*", font=("Arial", 12))
        cpE.grid(row=6, column=1, pady=5)
        cp_warn = tk.Label(frm, text="", bg="#2e2e2e", fg="orange", font=("Arial", 10))
        cp_warn.grid(row=7, column=1, sticky="w")
        def disable_paste(event): return "break"
        cpE.bind("<Control-v>", disable_paste)
        cpE.bind("<Control-V>", disable_paste)
        cpE.bind("<Shift-Insert>", disable_paste)
        cpE.bind("<<Paste>>", disable_paste)
        cpE.bind("<Button-3>", disable_paste)
        btn_frame = ttk.Frame(frm, style="TFrame")
        btn_frame.grid(row=8, column=0, columnspan=2, pady=(15, 0))
        def validate_fields():
            valid = True
            for w in [email_warn, name_warn, pw_warn, cp_warn]: w.config(text="")
            e, n, p, cp = emE.get().strip(), nmE.get().strip(), pwE.get(), cpE.get()
            if not e.lower().endswith("@gmail.com"):
                email_warn.config(text="Must be a valid @gmail.com address.")
                valid = False
            if not n.isalpha():
                name_warn.config(text="Letters only.")
                valid = False
            if len(p) < 8 or not p.isalnum():
                pw_warn.config(text="Min 8 chars; letters and numbers only.")
                valid = False
            if p != cp:
                cp_warn.config(text="Passwords do not match.")
                valid = False
            if e in self.users:
                email_warn.config(text="Email already registered.")
                valid = False
            return valid
        def save():
            if validate_fields():
                e, n, p = emE.get().strip(), nmE.get().strip(), pwE.get()
                self.db_manager.add_user(e, n, p, 'operator')
                self.users = self.db_manager.get_users()
                self.operator_listbox.insert(tk.END, f"{n} <{e}>")
                win.destroy()
        ttk.Button(btn_frame, text="Add", command=save).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).grid(row=0, column=1, padx=5)

    def delete_operator(self):
        sel = self.operator_listbox.curselection()
        if not sel:
            messagebox.showerror("Error", "Select operator.")
            return
        admin_pwd = simpledialog.askstring("Admin Confirmation", "Enter admin password:", show="*")
        if admin_pwd is None:
            return
        admin_user = next((u for u, info in self.users.items() if info['role'] == 'admin'), None)
        if not admin_user or self.users[admin_user]['password'] != admin_pwd:
            messagebox.showerror("Error", "Admin password incorrect.")
            return
        text = self.operator_listbox.get(sel)
        email = text.split("<")[1].strip(">")
        if messagebox.askyesno("Confirm", f"Delete operator {text}?"):
            self.db_manager.delete_user(email)
            self.users = self.db_manager.get_users()
            self.operator_listbox.delete(sel)

    def __del__(self):
        if hasattr(self, 'captures'):
            for cap in self.captures:
                try:
                    if cap.isOpened():
                        cap.release()
                except Exception:
                    pass
        if hasattr(self, 'db_manager'):
            self.db_manager.close()


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartMonitoringApp(root)
    root.mainloop()