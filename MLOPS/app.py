import os
import time
import random
import pickle
import pandas as pd
import numpy as np
import mysql.connector
import xgboost as xgb
from flask import Flask, render_template, request, flash
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_flask'

CATEGORY_MAP = {
    'misc_net': 0, 'grocery_pos': 1, 'entertainment': 2, 'gas_transport': 3,
    'misc_pos': 4, 'grocery_net': 5, 'shopping_net': 6, 'shopping_pos': 7,
    'food_dining': 8, 'personal_care': 9, 'health_fitness': 10, 'travel': 11,
    'kids_pets': 12, 'home': 13
}

FEATURE_COLUMNS = [
    'category',
    'lat',
    'long',
    'merch_lat',
    'merch_long',
    'amt',
    'time_last_trans',
    'lat_pre',
    'long_pre',
    'merch_lat_pre',
    'merch_long_pre',
    'amt_pre',
    'pre_mer'
]

def get_db_connection():
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', 'root')
    db_password = os.environ.get('DB_PASSWORD', '****')
    db_name = os.environ.get('DB_NAME', 'Transactions_Database')

    retries = 5
    while retries > 0:
        try:
            conn = mysql.connector.connect(
                host=db_host, user=db_user, password=db_password, database=db_name
            )
            return conn
        except mysql.connector.Error as err:
            print(f"[Database] Waiting... ({err})")
            time.sleep(3)
            retries -= 1
    raise Exception("DB Connection Failed")

def load_model_artifact():
    try:
        with open('model.pkl', 'rb') as f:
            artifact = pickle.load(f)
            if isinstance(artifact, dict) and 'model' in artifact:
                return artifact['model'], artifact.get('baseline_means', {})
            else:
                return artifact, {}
    except FileNotFoundError:
        return None, {}

global_model, global_baseline_means = load_model_artifact()

def calculate_online_features(cc_num, current_time_str, current_merchant):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT trans_date_trans_time, lat, `long`, merch_lat, merch_long, amt, merchant 
        FROM transactions 
        WHERE cc_num = %s 
        ORDER BY trans_date_trans_time DESC 
        LIMIT 1
    """
    cursor.execute(query, (cc_num,))
    last_trans = cursor.fetchone()
    cursor.close()
    conn.close()

    if last_trans:
        current_dt = datetime.strptime(current_time_str, '%Y-%m-%d %H-%M-%S')
        last_dt = pd.to_datetime(last_trans['trans_date_trans_time'])
        time_diff = (current_dt - last_dt).total_seconds()

        pre_mer_val = 1 if last_trans['merchant'] == current_merchant else 0
        
        return {
            'time_last_trans': abs(int(time_diff)),
            'lat_pre': float(last_trans['lat']),
            'long_pre': float(last_trans['long']),
            'merch_lat_pre': float(last_trans['merch_lat']),
            'merch_long_pre': float(last_trans['merch_long']),
            'amt_pre': float(last_trans['amt']),
            'pre_mer': pre_mer_val 
        }
    else:
        return {
            'time_last_trans': 0, 'lat_pre': 0, 'long_pre': 0, 
            'merch_lat_pre': 0, 'merch_long_pre': 0, 'amt_pre': 0,
            'pre_mer': 0
        }

def check_drift_and_retrain():
    global global_model, global_baseline_means
    
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    df.columns = [c.lower().strip() for c in df.columns]
    if len(df) < 50: return

    df['category'] = df['category'].map(CATEGORY_MAP).fillna(0)
    
    check_cols = ['amt', 'lat', 'long', 'merch_lat', 'merch_long', 
                  'time_last_trans', 'lat_pre', 'long_pre', 
                  'merch_lat_pre', 'merch_long_pre', 'amt_pre', 'pre_mer']

    drift_detected = False
    current_means = {}

    print("\n--- [MLOps] Checking Data Drift ---")
    for col in check_cols:
        col_name_pandas = 'long' if col == 'long' else col 
        if col_name_pandas not in df.columns: continue

        curr_mean = df[col_name_pandas].mean()
        current_means[col] = curr_mean
        
        base_mean = global_baseline_means.get(col, 0)
        if base_mean == 0: base_mean = 0.001
        
        change_pct = abs((curr_mean - base_mean) / base_mean)
        
        if change_pct > 0.3:
            print(f"!!! DRIFT DETECTED in '{col}': {change_pct*100:.2f}%")
            drift_detected = True
    
    if drift_detected:
        print(">>> Triggering Retraining...")
        try:
            X = df[FEATURE_COLUMNS]
            y = df['is_fraud']
            
            new_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            new_model.fit(X, y)
            
            global_model = new_model
            global_baseline_means = current_means
            
            artifact = {'model': new_model, 'baseline_means': current_means}
            with open('model.pkl', 'wb') as f:
                pickle.dump(artifact, f)
            print(">>> Retrained & Saved.")
        except Exception as e:
            print(f"ERROR Retraining: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT DISTINCT cc_num FROM transactions ORDER BY RAND() LIMIT 5")
    cc_nums = [row['cc_num'] for row in cursor.fetchall()]
    cursor.execute("SELECT DISTINCT merchant FROM transactions LIMIT 50")
    merchants = [row['merchant'] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    categories = list(CATEGORY_MAP.keys())

    if request.method == 'POST':
        try:
            cc_num = request.form['cc_num']
            merchant = request.form['merchant']
            category = request.form['category']
            amount = float(request.form['amount'])
            
            now = datetime.now()
            time_str = now.strftime('%Y-%m-%d %H-%M-%S')
            
            lat = random.uniform(-90, 90)
            long_val = random.uniform(-180, 180)
            merch_lat = random.uniform(-90, 90)
            merch_long = random.uniform(-180, 180)
            
            online_feats = calculate_online_features(cc_num, time_str, merchant)
            
            cat_encoded = CATEGORY_MAP.get(category, 0)
            
            input_data = pd.DataFrame([{
                'amt': amount,
                'lat': lat,
                'long': long_val,
                'merch_lat': merch_lat,
                'merch_long': merch_long,
                'category': cat_encoded,
                'time_last_trans': online_feats['time_last_trans'],
                'lat_pre': online_feats['lat_pre'],
                'long_pre': online_feats['long_pre'],
                'merch_lat_pre': online_feats['merch_lat_pre'],
                'merch_long_pre': online_feats['merch_long_pre'],
                'amt_pre': online_feats['amt_pre'],
                'pre_mer': online_feats['pre_mer'] 
            }])
            
            input_data = input_data[FEATURE_COLUMNS]
            
            if global_model:
                pred = global_model.predict(input_data)[0]
            else:
                pred = 0 
                
            conn = get_db_connection()
            cursor = conn.cursor()
            
            insert_query = """
            INSERT INTO transactions 
            (trans_date_trans_time, cc_num, merchant, category, amt, lat, `long`, merch_lat, merch_long,
             time_last_trans, lat_pre, long_pre, merch_lat_pre, merch_long_pre, amt_pre, pre_mer, is_fraud)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                time_str, cc_num, merchant, category, amount, lat, long_val, merch_lat, merch_long,
                online_feats['time_last_trans'], online_feats['lat_pre'], online_feats['long_pre'],
                online_feats['merch_lat_pre'], online_feats['merch_long_pre'], online_feats['amt_pre'],
                online_feats['pre_mer'], 
                int(pred)
            ))
            conn.commit()
            cursor.close()
            conn.close()
            
            if pred == 1:
                flash("This is a fraud transaction", "danger")
            else:
                flash("The money was sent succesfully", "success")
                
            check_drift_and_retrain()
            
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            print(f"DEBUG: {e}")

    return render_template('index.html', cc_nums=cc_nums, merchants=merchants, categories=categories)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

