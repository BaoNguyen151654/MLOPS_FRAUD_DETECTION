import mysql.connector
import os
import time

def get_db_connection():
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', 'root')
    db_password = os.environ.get('DB_PASSWORD', 'giabao04052000') 
    db_name = os.environ.get('DB_NAME', 'Transactions_Database')

    retries = 5
    while retries > 0:
        try:
            print(f"Connecting to Database at {db_host}...")
            conn = mysql.connector.connect(
                host=db_host,
                user=db_user,
                password=db_password,
                database=db_name
            )
            print("Database connected successfully!")
            return conn
        except mysql.connector.Error as err:
            time.sleep(3)
            retries -= 1
            
    raise Exception("The app is death =))")