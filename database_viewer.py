import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox

class SmartMonitoringApp:
    def __init__(self, root):
        self.root = root
        self.db_path = "camera_reports/monitoring.db"  # Adjust path as needed
        self.current_theme = "light"  # Default theme
        self.style = ttk.Style()
        self.create_widgets()

    def create_widgets(self):
        # Report frame setup
        self.report_frame = ttk.Frame(self.root)
        self.report_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add buttons to view database tables
        button_frame = ttk.Frame(self.report_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(button_frame, text="View Anomaly Status", command=lambda: self.show_table_contents("Anomaly_status")).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="View Login Table", command=lambda: self.show_table_contents("login")).pack(side=tk.LEFT, padx=5)

    def show_table_contents(self, table_name):
        """Display the contents of the specified table in a new GUI window."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name}")
            records = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            conn.close()
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"Error accessing database: {e}")
            return

        # Create a new Toplevel window
        db_window = tk.Toplevel(self.root)
        db_window.title(f"{table_name} Table Contents")
        db_window.geometry("1000x600")

        # Create a Treeview widget
        tree = ttk.Treeview(db_window, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)

        # Insert records into the Treeview
        for record in records:
            tree.insert("", "end", values=record)

        tree.pack(fill=tk.BOTH, expand=True)

        # Add a close button
        close_button = ttk.Button(db_window, text="Close", command=db_window.destroy)
        close_button.pack(pady=10)

        # Apply theme to the new window
        self.apply_theme_to_window(db_window)

    def apply_theme_to_window(self, window):
        """Apply the current theme to the new window."""
        if self.current_theme == "dark":
            bg_color = "black"
            fg_color = "white"
        else:
            bg_color = "white"
            fg_color = "black"
        window.configure(bg=bg_color)
        self.style.configure("Treeview", background=bg_color, foreground=fg_color, fieldbackground=bg_color)
        self.style.configure("Treeview.Heading", background=bg_color, foreground=fg_color)

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartMonitoringApp(root)
    root.mainloop()