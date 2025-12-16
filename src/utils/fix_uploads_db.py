import sqlite3

# Connect to the existing uploads.db file
conn = sqlite3.connect('uploads.db')
c = conn.cursor()

# Create the 'uploads' table if it doesn't already exist
c.execute('''
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        prediction TEXT NOT NULL,
        report TEXT,
        date TEXT NOT NULL
    )
''')

conn.commit()
conn.close()

print("Fixed: 'uploads' table has been created in uploads.db.")
