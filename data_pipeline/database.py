import os
import sqlite3

class DataBase:
    
    def __init__(self):
        # Calculate the dynamic workspace path once during object initialization
        module_dir = os.path.dirname(os.path.abspath(__file__))
        db_folder = os.path.join(module_dir, 'database')
        self.db_path = os.path.join(db_folder, 'books.db')
    
    def create_conn_tables_database(self):
        # 1. Safely check and log file existence
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if os.path.exists(self.db_path):
            print(f"Database file found at: {self.db_path}")
        else:
            print(f"No existing database found. Creating a new file at: {self.db_path}")
    
        # 2. Establish context-managed connection using the correct absolute path
        with sqlite3.connect(self.db_path) as connection:
            # Enable Foreign Key constraints in SQLite explicitly
            connection.execute("PRAGMA foreign_keys = ON;")

            # 3. Create categories table
            connection.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category_name TEXT UNIQUE NOT NULL
                )
            """)
            
            # 4. Create books table with fixed keys and schema definitions
            connection.execute("""
                CREATE TABLE IF NOT EXISTS books (
                    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    price_inr REAL,
                    price_gbp REAL,
                    rating REAL,
                    availability INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(category_id) REFERENCES categories (category_id) ON DELETE CASCADE
                )
            """)
            
            # 'with' statement handles commits automatically on success, 
            # but explicit commit keeps the pipeline explicit
            connection.commit()
            
        print("Database schema verified and tables ready successfully.")