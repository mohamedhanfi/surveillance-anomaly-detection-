import sqlite3
import os
import threading

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.db_lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.init_database()

    def init_database(self):
        with self.db_lock:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS User (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT,
                    username TEXT UNIQUE,
                    password TEXT,
                    role TEXT
                )
            ''')
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS Anomaly_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Timestamp TEXT,
                    Predicted_class TEXT,
                    Cam_Num TEXT,
                    Confidence REAL,
                    report TEXT
                )
            ''')
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anomaly_id INTEGER,
                    snapshot_path TEXT,
                    caption TEXT,
                    FOREIGN KEY (anomaly_id) REFERENCES Anomaly_status(id)
                )
            ''')
            self.cursor.execute("SELECT COUNT(*) FROM User WHERE role = 'admin'")
            if self.cursor.fetchone()[0] == 0:
                self.cursor.execute('''
                    INSERT INTO User (email, username, password, role)
                    VALUES ('admin@gmail.com', 'Admin', '123456789', 'admin')
                ''')
            self.conn.commit()

    def insert_anomaly(self, timestamp, predicted_class, cam_num, confidence, report):
        with self.db_lock:
            self.cursor.execute('''
                INSERT INTO Anomaly_status (Timestamp, Predicted_class, Cam_Num, Confidence, report)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, predicted_class, cam_num, confidence, report))
            anomaly_id = self.cursor.lastrowid
            self.conn.commit()
            return anomaly_id

    def insert_snapshot(self, anomaly_id, snapshot_path, caption):
        with self.db_lock:
            self.cursor.execute('''
                INSERT INTO snapshots (anomaly_id, snapshot_path, caption)
                VALUES (?, ?, ?)
            ''', (anomaly_id, snapshot_path, caption))
            self.conn.commit()

    def get_anomaly(self, anomaly_id):
        with self.db_lock:
            self.cursor.execute("SELECT * FROM Anomaly_status WHERE id = ?", (anomaly_id,))
            return self.cursor.fetchone()

    def get_snapshots(self, anomaly_id):
        with self.db_lock:
            self.cursor.execute("SELECT snapshot_path, caption FROM snapshots WHERE anomaly_id = ?", (anomaly_id,))
            return self.cursor.fetchall()

    def get_users(self):
        with self.db_lock:
            self.cursor.execute("SELECT email, username, password, role FROM User")
            users = self.cursor.fetchall()
            return {email: {'name': username, 'password': password, 'role': role} for email, username, password, role in users}

    def add_user(self, email, username, password, role):
        with self.db_lock:
            
            self.cursor.execute(''' 
                INSERT INTO User (email, username, password, role)
                VALUES (?, ?, ?, ?)
            ''', (email, username, password, role))
            self.conn.commit()

    def delete_user(self, email):
        with self.db_lock:
            self.cursor.execute("DELETE FROM User WHERE email = ?", (email,))
            self.conn.commit()



    def close(self):
        self.conn.close()
        
        
        
        