import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv

CLASS_NAMES = ["Arson", "Fighting", "RoadAccidents", "Robbery", "Normal"]

class SmartMonitoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Monitoring & Anomaly Detection – Database Viewer")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        self.db_path = "camera_reports/monitoring.db"
        self.rows_per_page = 50
        self.current_page = 1
        
        # Modern color palette
        self.theme_colors = {
            "light": {
                "bg": "#F9FAFC",
                "fg": "#1A202C",
                "accent": "#4FD1C5",  # Teal accent
                "card": "#FFFFFF",
                "border": "#CBD5E0",
                "odd_row": "#F7FAFC",  # Light blue wash
                "even_row": "#FFFFFF",
                "header": "#EDF2F7"
            },
            "dark": {
                "bg": "#1A202C",
                "fg": "#E2E8F0",
                "accent": "#48BB78",  # Emerald green
                "card": "#2D3748",
                "border": "#4A5568",
                "odd_row": "#2C3545",
                "even_row": "#1F2737",
                "header": "#2C3545"
            }
        }
        
        self.current_theme = "light"
        self.style = ttk.Style()
        self._setup_modern_styles()

        # Create notebook interface
        notebook = ttk.Notebook(self.root, style="Modern.TNotebook")
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Initialize tabs
        self._build_table_viewer(ttk.Frame(notebook))
        self._build_user_logins(ttk.Frame(notebook))
        self._build_stats_tab(ttk.Frame(notebook))

        # Add tabs to notebook with emoji icons
        notebook.add(self.table_tab, text="📊 Table Viewer")
        notebook.add(self.login_tab, text="🔐 User Logins")
        notebook.add(self.stats_tab, text="📈 Statistics")

        # Load initial data
        self.table_names = self._get_table_names()
        if self.table_names:
            self.table_var.set(self.table_names[0])
            self.load_table_page()

        # Apply theme
        self.toggle_theme(initial=True)

    def _setup_modern_styles(self):
        """Configure modern UI styles for all widgets"""
        colors = self.theme_colors[self.current_theme]

        # Main window background
        self.root.configure(bg=colors["bg"])

        # Notebook styling
        self.style.configure("Modern.TNotebook", background=colors["card"])
        self.style.configure("Modern.TNotebook.Tab",
                            padding=[15, 8],
                            font=("Segoe UI", 10, "bold"),
                            background=colors["card"],
                            foreground=colors["fg"])
        self.style.map("Modern.TNotebook.Tab",
                      background=[("selected", colors["accent"]),
                                 ("active", self._lighten_color(colors["accent"], 0.1))],
                      foreground=[("selected", "#FFFFFF")])

        # Button styling with hover effect
        self.style.configure("Modern.TButton",
                            font=("Segoe UI", 9, "bold"),
                            padding=10,
                            relief="flat",
                            borderwidth=0,
                            background=colors["accent"],
                            foreground="#FFFFFF")
        self.style.map("Modern.TButton",
                      background=[("pressed", self._darken_color(colors["accent"], 0.1)),
                                 ("active", self._lighten_color(colors["accent"], 0.1))])

        # Entry fields
        self.style.configure("Modern.TEntry",
                            padding=8,
                            fieldbackground=colors["header"],
                            bordercolor=colors["border"])

        # Label styling
        self.style.configure("Modern.TLabel",
                            background=colors["bg"],
                            foreground=colors["fg"],
                            font=("Segoe UI", 10))

        # Frame styling
        self.style.configure("Modern.TFrame", background=colors["bg"])

        # Treeview improvements
        self.style.configure("Modern.Treeview.Heading",
                            font=("Segoe UI", 10, "bold"),
                            background=colors["header"],
                            foreground=colors["fg"],
                            relief="flat")
        self.style.configure("Modern.Treeview",
                            background=colors["card"],
                            fieldbackground=colors["bg"],
                            foreground=colors["fg"],
                            rowheight=30,
                            borderwidth=0)
        self.style.map("Modern.Treeview.Heading",
                      relief=[("pressed", "!disabled", "sunken"),
                             ("!pressed", "!disabled", "flat")])

    def toggle_theme(self, initial=False):
        """Switch between light and dark themes with modern styling"""
        if not initial:
            self.current_theme = "dark" if self.current_theme == "light" else "light"
            self.theme_btn.config(text="☀️ Light Mode" if self.current_theme == "dark" else "🌙 Dark Mode")

        colors = self.theme_colors[self.current_theme]
        self._setup_modern_styles()

        # Update treeview row tags
        for tree in (getattr(self, 'table_tree', None), getattr(self, 'login_tree', None)):
            if tree:
                tree.tag_configure("oddrow", background=colors["odd_row"])
                tree.tag_configure("evenrow", background=colors["even_row"])

    def _build_table_viewer(self, parent):
        self.table_tab = parent
        top_frame = ttk.Frame(parent, style="Modern.TFrame")
        top_frame.pack(fill="x", pady=(5, 10), padx=10)

        control_frame = ttk.Frame(top_frame, style="Modern.TFrame")
        control_frame.pack(fill="x")

        ttk.Label(control_frame, text="🔍 Select Table:", style="Modern.TLabel").pack(side="left", padx=5)
        self.table_names = self._get_table_names()
        self.table_var = tk.StringVar(value=self.table_names[0] if self.table_names else "")
        self.table_combo = ttk.Combobox(control_frame, textvariable=self.table_var,
                                       values=self.table_names, state="readonly", width=25)
        self.table_combo.pack(side="left", padx=5)
        self.table_combo.bind("<<ComboboxSelected>>", lambda e: self.load_table_page())

        ttk.Button(control_frame, text="⟳ Refresh", command=self.load_table_page,
                  style="Modern.TButton").pack(side="left", padx=5)
        ttk.Button(control_frame, text="📁 Export CSV", command=self.export_table,
                  style="Modern.TButton").pack(side="right", padx=5)

        ttk.Label(control_frame, text="🔍 Filter:", style="Modern.TLabel").pack(side="left", padx=(20, 0))
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(control_frame, textvariable=self.search_var)
        self.search_entry.pack(side="left", padx=5, fill="x", expand=True)
        self.search_entry.bind("<KeyRelease>", lambda e: self.apply_filter())

        self.theme_btn = ttk.Button(control_frame, text="🌙 Dark Mode",
                                 command=self.toggle_theme, style="Modern.TButton")
        self.theme_btn.pack(side="right", padx=5)

        table_frame = ttk.Frame(parent, style="Modern.TFrame")
        table_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.table_tree = ttk.Treeview(table_frame, show="headings", style="Modern.Treeview")
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.table_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table_tree.xview)
        self.table_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.table_tree.pack(fill="both", expand=True)

        # Pagination controls
        pagination_frame = ttk.Frame(parent, style="Modern.TFrame")
        pagination_frame.pack(fill="x", padx=12, pady=5)
        self.prev_btn = ttk.Button(pagination_frame, text="◄ Prev", 
                                 command=lambda: self.change_page(-1), style="Modern.TButton")
        self.prev_btn.pack(side="left", padx=5)
        self.page_label = ttk.Label(pagination_frame, text="Page: 1", style="Modern.TLabel")
        self.page_label.pack(side="left")
        self.next_btn = ttk.Button(pagination_frame, text="Next ►", 
                                 command=lambda: self.change_page(1), style="Modern.TButton")
        self.next_btn.pack(side="left", padx=5)

        # Status bar
        status_frame = ttk.Frame(parent, style="Modern.TFrame")
        status_frame.pack(side="bottom", fill="x", ipady=2)
        self.status_var = tk.StringVar(value="Ready - No table loaded")
        ttk.Label(status_frame, textvariable=self.status_var, style="Modern.TLabel", anchor="w", padding=8).pack(fill="x")

    def _build_user_logins(self, parent):
        self.login_tab = parent
        ttk.Label(parent, text="🔐 User Login Records", style="Modern.TLabel",
                 font=("Segoe UI", 14, "bold")).pack(pady=(10, 5))

        card = ttk.Frame(parent, style="Modern.TFrame")
        card.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        ctrl = ttk.Frame(card, style="Modern.TFrame")
        ctrl.pack(fill="x", pady=5, padx=5)
        ttk.Button(ctrl, text="📁 Export CSV", command=self.export_logins,
                 style="Modern.TButton").pack(side="right", padx=5)
        ttk.Button(ctrl, text="⟳ Refresh", command=self.load_user_logins,
                 style="Modern.TButton").pack(side="right", padx=5)

        frame = ttk.Frame(card, style="Modern.TFrame")
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.login_tree = ttk.Treeview(frame, show="headings", style="Modern.Treeview")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.login_tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.login_tree.xview)
        self.login_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.login_tree.pack(fill="both", expand=True)

        self.load_user_logins()

    def _build_stats_tab(self, parent):
        self.stats_tab = parent
        ctrl = ttk.Frame(parent, style="Modern.TFrame")
        ctrl.pack(fill="x", padx=10, pady=8)

        ttk.Button(ctrl, text="⟳ Refresh", command=self.load_stats,
                  style="Modern.TButton").pack(side="left", padx=5)
        ttk.Label(ctrl, text="📅 Select Day (YYYY-MM-DD):", style="Modern.TLabel").pack(
            side="left", padx=(20, 0))
        self.day_var = tk.StringVar(value=datetime.today().strftime("%Y-%m-%d"))
        ttk.Entry(ctrl, textvariable=self.day_var, width=12, style="Modern.TEntry").pack(
            side="left", padx=5)
        self.stats_status_var = tk.StringVar(value="Last refreshed: Never")
        ttk.Label(ctrl, textvariable=self.stats_status_var, style="Modern.TLabel",
                 anchor="e").pack(side="right", expand=True, fill="x", padx=10)

        plots = ttk.Frame(parent, style="Modern.TFrame")
        plots.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Initialize figures and canvases
        self.fig1, self.ax1 = plt.subplots(figsize=(10, 4), dpi=100)
        self.fig1.set_facecolor(self.theme_colors[self.current_theme]["card"])
        self.ax1.set_facecolor(self.theme_colors[self.current_theme]["card"])
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plots)
        self.canvas1.get_tk_widget().pack(fill="both", expand=True, pady=5)

        self.fig2, self.ax2 = plt.subplots(figsize=(10, 4), dpi=100)
        self.fig2.set_facecolor(self.theme_colors[self.current_theme]["card"])
        self.ax2.set_facecolor(self.theme_colors[self.current_theme]["card"])
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plots)
        self.canvas2.get_tk_widget().pack(fill="both", expand=True, pady=5)

        self.fig3, self.ax3 = plt.subplots(figsize=(10, 4), dpi=100)
        self.fig3.set_facecolor(self.theme_colors[self.current_theme]["card"])
        self.ax3.set_facecolor(self.theme_colors[self.current_theme]["card"])
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=plots)
        self.canvas3.get_tk_widget().pack(fill="both", expand=True, pady=5)

        self.load_stats()

    def _get_table_names(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
                return [r[0] for r in cur.fetchall()]
        except sqlite3.Error as e:
            messagebox.showerror("Error", f"Cannot read DB: {e}")
            return []

    def load_table_page(self):
        tbl = self.table_var.get()
        if not tbl:
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                offset = (self.current_page - 1) * self.rows_per_page
                
                # Get filtered count
                search_term = self.search_var.get().lower()
                cols = self.get_table_columns(tbl)
                
                where_clause = ""
                if search_term:
                    # build a list of individual conditions, e.g. ["col1 LIKE '%foo%'", "col2 LIKE '%foo%'", …]
                    patterns = [f"{col} LIKE '%{search_term}%'" for col in cols]
                    # join them with OR
                    where_clause = "WHERE " + " OR ".join(patterns)

                cur.execute(f"SELECT COUNT(*) FROM {tbl} {where_clause}")
                total_rows = cur.fetchone()[0]
                
                cur.execute(f"SELECT * FROM {tbl} {where_clause} LIMIT {self.rows_per_page} OFFSET {offset}")
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                
        except sqlite3.Error as e:
            messagebox.showerror("Error", f"Failed to load {tbl}: {e}")
            return
        
        self.update_table_display(cols, rows)
        self.update_pagination_controls(total_rows)
        self.status_var.set(f"Showing {len(rows)} of {total_rows} rows from '{tbl}'")

    def get_table_columns(self, table_name):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute(f"PRAGMA table_info({table_name})")
                return [row[1] for row in cur.fetchall()]
        except sqlite3.Error:
            return []

    def update_table_display(self, cols, rows):
        self.table_tree.delete(*self.table_tree.get_children())
        self.table_tree["columns"] = cols
        
        for col in cols:
            self.table_tree.heading(col, text=col, command=lambda c=col: self.sort_column(c))
            self.table_tree.column(col, width=120, anchor="w")
        
        for i, row in enumerate(rows):
            tag = "evenrow" if i % 2 == 0 else "oddrow"
            self.table_tree.insert("", "end", values=row, tags=(tag,))

    def update_pagination_controls(self, total_rows):
        total_pages = max(1, (total_rows + self.rows_per_page - 1) // self.rows_per_page)
        self.current_page = max(1, min(self.current_page, total_pages))
        self.page_label.config(text=f"Page: {self.current_page}/{total_pages}")
        self.prev_btn.config(state="normal" if self.current_page > 1 else "disabled")
        self.next_btn.config(state="normal" if self.current_page < total_pages else "disabled")

    def change_page(self, delta):
        self.current_page += delta
        self.load_table_page()

    def sort_column(self, col):
        data = [(self.table_tree.set(child, col), child) for child in self.table_tree.get_children("")]
        data.sort(key=lambda x: x[0])
        for index, (_, iid) in enumerate(data):
            self.table_tree.move(iid, "", index)
            self.table_tree.item(iid, tags=("evenrow" if index % 2 == 0 else "oddrow",))

    def apply_filter(self, event=None):
        self.current_page = 1
        self.load_table_page()

    def export_table(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not filename:
            return

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.table_tree["columns"])
                
                for item in self.table_tree.get_children():
                    if self.table_tree.parent(item) == '':  # Only visible rows
                        values = self.table_tree.item(item)['values']
                        writer.writerow([str(v) if v is not None else '' for v in values])
            
            messagebox.showinfo("Success", "Table data exported successfully!")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{e}")

    def load_user_logins(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM login")
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
        except sqlite3.Error as e:
            messagebox.showerror("Error", f"Failed to load login data: {e}")
            return
        
        self.login_tree.delete(*self.login_tree.get_children())
        self.login_tree["columns"] = cols
        
        for col in cols:
            self.login_tree.heading(col, text=col)
            self.login_tree.column(col, width=150, anchor="w")
        
        for i, row in enumerate(rows):
            tag = "evenrow" if i % 2 == 0 else "oddrow"
            self.login_tree.insert("", "end", values=row, tags=(tag,))

    def export_logins(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not filename:
            return

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.login_tree["columns"])
                
                for item in self.login_tree.get_children():
                    if self.login_tree.parent(item) == '':  # Only visible rows
                        values = self.login_tree.item(item)['values']
                        writer.writerow([str(v) if v is not None else '' for v in values])
            
            messagebox.showinfo("Success", "Login data exported successfully!")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export login data:\n{e}")

    def load_stats(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT predicted_class, timestamp FROM Anomaly_status")
                rows = cur.fetchall()
        except sqlite3.Error:
            rows = []

        monthly = {}
        for cls, ts in rows:
            month = ts.split("_")[0][:7] if ts else ""
            if month:
                monthly.setdefault(month, {}).setdefault(cls, 0)
                monthly[month][cls] += 1
                monthly[month]["_total"] = monthly[month].get("_total", 0) + 1

        months = sorted(monthly.keys())
        classes = sorted(set(c for m in monthly for c in monthly[m] if c != "_total"))

        # Plot monthly per-class
        self.ax1.clear()
        for cls in classes:
            y = [monthly[m].get(cls, 0) for m in months]
            self.ax1.plot(months, y, label=cls)
        self.ax1.set_title("Anomalies per Class by Month", color=self.theme_colors[self.current_theme]["fg"])
        self.ax1.set_ylabel("Count")
        self.ax1.set_xticks(range(len(months)))
        self.ax1.set_xticklabels(months, rotation=45, ha="right")
        self.ax1.legend()
        self.fig1.tight_layout()
        self.canvas1.draw()

        # Plot monthly totals
        self.ax2.clear()
        if months:
            tot = [monthly[m]["_total"] for m in months]
            self.ax2.plot(months, tot)
        self.ax2.set_title("Total Events per Month", color=self.theme_colors[self.current_theme]["fg"])
        self.ax2.set_ylabel("Count")
        self.ax2.set_xticks(range(len(months)))
        self.ax2.set_xticklabels(months, rotation=45, ha="right")
        self.fig2.tight_layout()
        self.canvas2.draw()

        # Plot daily per-hour
        day = self.day_var.get().strip()
        hourly = {h: 0 for h in range(24)}
        for cls, ts in rows:
            if ts and "_" in ts:
                datepart, timepart = ts.split("_")
                if datepart == day:
                    hour = int(timepart.split("-")[0]) if "-" in timepart else 0
                    hourly[hour] += 1

        self.ax3.clear()
        hours = list(range(24))
        counts = [hourly[h] for h in hours]
        self.ax3.plot(hours, counts)
        self.ax3.set_title(f"Anomalies on {day} by Hour", color=self.theme_colors[self.current_theme]["fg"])
        self.ax3.set_xlabel("Hour of Day")
        self.ax3.set_ylabel("Count")
        self.ax3.set_xticks(hours)
        self.fig3.tight_layout()
        self.canvas3.draw()
        
        # Update status
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stats_status_var.set(f"Last refreshed: {now}")

    def _lighten_color(self, color, factor=0.1):
        """Lighten a hex color by a given factor"""
        r = int(color[1:3], 16) * (1 + factor)
        g = int(color[3:5], 16) * (1 + factor)
        b = int(color[5:7], 16) * (1 + factor)
        return f"#{int(min(r, 255)):02X}{int(min(g, 255)):02X}{int(min(b, 255)):02X}"

    def _darken_color(self, color, factor=0.1):
        """Darken a hex color by a given factor"""
        r = int(color[1:3], 16) * (1 - factor)
        g = int(color[3:5], 16) * (1 - factor)
        b = int(color[5:7], 16) * (1 - factor)
        return f"#{int(max(r, 0)):02X}{int(max(g, 0)):02X}{int(max(b, 0)):02X}"

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartMonitoringApp(root)
    root.mainloop()