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
import json

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
            'pre_mer': 0}

STATE_FILE = 'mlops_state.json'
def get_last_check_count(current_count):
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'w') as f:
            json.dump({'last_count': current_count}, f)
        return current_count
    try:
        with open(STATE_FILE, 'r') as f:
            data = json.load(f) 
            return data.get('last_count', 0) 
    except Exception: 
        print(">>> [Warning] State file is empty/corrupt. Resetting...")
        with open(STATE_FILE, 'w') as f:
            json.dump({'last_count': current_count}, f) 
        return current_count
def update_last_check_count(new_count):
    with open(STATE_FILE, 'w') as f:
        json.dump({'last_count': new_count}, f)

def calculate_psi(expected, actual, buckets=10):
    def scale_range(input, min_val, max_val):
        input = input.copy() 
        input += -(np.min(input))
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 0.0001 
        current_max = np.max(input)
        if current_max == 0:
            current_max = 0.0001   
        input /= (current_max / range_val)
        input += min_val
        return input
    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    def sub_psi(e_perc, a_perc):
        if a_perc == 0: a_perc = 0.0001
        if e_perc == 0: e_perc = 0.0001
        return (e_perc - a_perc) * np.log(e_perc / a_perc)
    psi_value = np.sum([sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))])
    return psi_value

def check_drift_and_retrain():
    global global_model 
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM transactions ORDER BY id ASC", conn)
    conn.close()
    df.columns = [c.lower().strip() for c in df.columns]
    if 'category' in df.columns:
        df['category'] = df['category'].map(CATEGORY_MAP).fillna(0)
    current_total_rows = len(df)
    last_check_count = get_last_check_count(current_total_rows)
    trigger_threshold = last_check_count * 1.2
    print(f"\n--- [MLOps Status] Current: {current_total_rows} | Last Check: {last_check_count} | Target: {int(trigger_threshold)} ---")
    if current_total_rows < trigger_threshold:
        return
    print(">>> DATA GROWTH > 20% DETECTED. INITIATING DRIFT CHECK...")

    split_idx = int(current_total_rows * 0.8)
    baseline_df = df.iloc[:split_idx]
    current_df = df.iloc[split_idx:]
    print(f"   -> Splitting Data: Reference ({len(baseline_df)}) vs New Data ({len(current_df)})")

    check_cols = ['amt', 'lat', 'long', 'merch_lat', 'merch_long', 
                  'time_last_trans', 'lat_pre', 'long_pre', 
                  'merch_lat_pre', 'merch_long_pre', 'amt_pre']
    total_psi = 0
    valid_cols = 0
    
    for col in check_cols:
        if col not in df.columns: continue
        try:
            psi = calculate_psi(baseline_df[col].values, current_df[col].values)
            total_psi += psi
            valid_cols += 1
        except: continue      
    avg_psi = total_psi / valid_cols if valid_cols > 0 else 0
    print(f"   -> Average PSI: {avg_psi:.4f} (Threshold: 0.3)")
    
    if avg_psi > 0.3:
        print(">>> DRIFT ALERT (PSI > 0.3). EXECUTING BLIND RETRAIN...")
        try:
            X = df[FEATURE_COLUMNS]
            y = df['is_fraud']
            
            new_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            new_model.fit(X, y)

            global_model = new_model
            artifact = {'model': new_model}
            with open('model.pkl', 'wb') as f:
                pickle.dump(artifact, f)           
            print(">>> MODEL RETRAINED & SAVED SUCCESSFULY.")
        except Exception as e:
            print(f"ERROR Retraining: {e}")
    else:
        print(">>> DATA STABLE (PSI <= 0.3). NO RETRAIN NEEDED.")
    update_last_check_count(current_total_rows)
    print(">>> CHECKPOINT UPDATED. Waiting for next 20% data...")

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

