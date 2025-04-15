import os
import pandas as pd
import uuid
import json
import decimal
import platform
import time
import base64
import re
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import importlib
importlib.reload(plt)
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import pymysql
from notebook_logger import append_to_notebook
import mysql.connector as mysql  # For mysql-connector-python
from sqlalchemy import create_engine
from pyhive import hive
from together import Together
import together
from spellchecker import SpellChecker
import sqlparse

import os
import uuid
import joblib
import pandas as pd
import traceback
from flask import Blueprint, request, jsonify, send_file
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.preprocessing import StandardScaler
import shap
from io import BytesIO
from datetime import datetime
from flask import Blueprint, request, jsonify


UPLOAD_FOLDER = '/tmp/uploads'
MODEL_DIR = '/===tmp/saved_models'
NOTEBOOK_DIR = '/tmp/notebooks'
SESSION_DIR = '/tmp/sessions'

# ✅ Make sure folders exist
for path in [UPLOAD_FOLDER, MODEL_DIR, NOTEBOOK_DIR, SESSION_DIR]:
    os.makedirs(path, exist_ok=True)

os.makedirs('/tmp/uploads', exist_ok=True)
os.makedirs('/tmp/saved_models', exist_ok=True)
os.makedirs('/tmp/notebooks', exist_ok=True)
os.makedirs('/tmp/sessions', exist_ok=True)


client = Together()

app = Flask(__name__)
CORS(app)  #
app.secret_key = "123456"

# ✅ Configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Together.AI Configuration
TOGETHER_API_KEY = "6f162109add786c0563504e18939cbd0e04e176acb4bf83c5c3113213312f84d"
together.api_key = TOGETHER_API_KEY

# ✅ Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Persistent session storage
# --------- Helper Functions ---------

def get_session_path(file_id):
    return os.path.join('/tmp/sessions', f"{file_id}.json")

def load_session(file_id):
    try:
        with open(get_session_path(file_id), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_session(file_id, data):
    with open(get_session_path(file_id), "w") as f:
        json.dump(data, f)

def serialize_data(rows):
    """Converts MySQL result rows with Decimal to float."""
    serialized = []

    for row in rows:
        serialized_row = {}
        for key, value in row.items():
            if isinstance(value, decimal.Decimal):
                serialized_row[key] = float(value)  # Convert Decimal to float
            else:
                serialized_row[key] = value
        serialized.append(serialized_row)

    return serialized


@app.route('/rollback/<file_id>', methods=['POST'])
def rollback_changes(file_id):
    print(f"🔁 Step-by-step rollback request for: {file_id}")

    # ✅ Load user's session data
    session_data = load_session(file_id)

    if not session_data:
        return jsonify({"success": False, "error": "File not found"}), 404

    try:
        # ✅ Ensure history exists
        if not session_data.get('history'):
            print("⚠️ No more steps to rollback.")
            data_json = pd.DataFrame(session_data['modified']).fillna("").astype(str).to_dict(orient='records')
            return jsonify({
                "success": True,
                "info": "No more steps to rollback.",
                "data": data_json
            })

        # ✅ Rollback one step
        session_data['modified'] = session_data['history'].pop()

        # ✅ Save updated session
        save_session(file_id, session_data)

        # ✅ Convert DataFrame to JSON-safe structure
        data_json = pd.DataFrame(session_data['modified']).fillna("").astype(str).to_dict(orient='records')

        print("✅ Rolled back successfully.")
        return jsonify({"success": True, "data": data_json})

    except Exception as e:
        print(f"❌ Exception during rollback: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500




class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)  # Convert Decimal to float
        return super(DecimalEncoder, self).default(obj)


@app.route('/connect', methods=['POST'])
def connect_db():
    """Connect to MySQL database and store connection in session file."""
    try:
        data = request.get_json()
        db_type = data.get('dbType')
        host = data.get('host')
        port = int(data.get('port'))
        username = data.get('username')
        password = data.get('password')
        database = data.get('database')
        table = data.get('table')

        if db_type != "sql":
            return jsonify({"success": False, "error": "Unsupported database type"})

        # ✅ Establish connection
        connection = pymysql.connect(
            host=host,
            user=username,
            password=password,
            database=database,
            port=port,
            cursorclass=pymysql.cursors.DictCursor  # Return results as dictionaries
        )

        with connection.cursor() as cursor:
            # ✅ Preview first 100 rows
            preview_query = f"SELECT * FROM {table} LIMIT 100"
            cursor.execute(preview_query)
            rows = cursor.fetchall()

            # ✅ Convert Decimal to float in the query result
            for row in rows:
                for key, value in row.items():
                    if isinstance(value, decimal.Decimal):
                        row[key] = float(value)

            columns = [desc[0] for desc in cursor.description]

        connection.close()

        # ✅ Store connection details & table data in session
        df = pd.DataFrame(rows, columns=columns)
        file_id = str(uuid.uuid4())

        session_data = {
            'db_type': 'sql',
            'host': host,
            'port': port,
            'username': username,
            'password': password,
            'database': database,
            'table': table,
            'original': df.to_dict(),   # ✅ Store as dict (JSON-safe)
            'modified': df.to_dict(),   # ✅ Store as dict (JSON-safe)
            'history': []               # ✅ Initialize empty history
        }

        # ✅ Save session to /tmp/sessions/file_id.json
        save_session(file_id, session_data)

        # ✅ Generate schema for SQL table
        generate_schema(file_id, df)

        return jsonify({
            "success": True,
            "file_id": file_id,
            "redirect_url": url_for('describe_file', file_id=file_id)
        })

    except Exception as e:
        print(f"❌ Error connecting to database: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500



@app.route('/modeldev/<file_id>', methods=['GET'])
def model_development_page(file_id):
    try:
        # ✅ Load user's session data
        session_data = load_session(file_id)

        if not session_data:
            return "Invalid file ID", 404

        df = pd.DataFrame(session_data.get('modified', {}))  # Convert dict back to DataFrame

        if df.empty:
            return "Modified dataset not found", 404

        return render_template("modeldev.html", file_id=file_id)

    except Exception as e:
        print(f"❌ Error loading model dev page: {e}")
        traceback.print_exc()
        return "Error loading model development page", 500


@app.route('/save/<file_id>', methods=['POST'])
def save_file(file_id):
    try:
        filename = request.json.get('filename', 'cleaned_data.csv')

        # ✅ Load user's session data
        session_data = load_session(file_id)

        if not session_data:
            return jsonify({"success": False, "error": "File not found"}), 404

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

        # ✅ Convert dict back to DataFrame
        df = pd.DataFrame(session_data.get('modified', {}))

        if df.empty:
            return jsonify({"success": False, "error": "Modified data is empty"}), 400

        # ✅ Save file
        df.to_csv(file_path, index=False)

        return jsonify({"success": True, "message": "File saved successfully", "file_path": file_path})

    except Exception as e:
        print(f"❌ Error saving file: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Failed to save file: {str(e)}"}), 500



def sanitize_filename(filename):
    sanitized = secure_filename(filename)
    max_length = 100
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:max_length - len(ext)] + ext

    unique_id = str(uuid.uuid4())[:8]
    sanitized = f"{unique_id}_{sanitized}"
    sanitized = os.path.normpath(sanitized)

    print(f"✅ Sanitized filename: {sanitized}")
    return sanitized


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_nan_to_none(obj):
    """Recursively converts NaN, Infinity, and -Infinity to None for JSON compatibility."""
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(x) for x in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    else:
        return obj


def generate_schema(file_id, df):
    try:
        if not isinstance(file_id, str) or not file_id.strip():
            raise ValueError("Invalid file ID provided for schema generation")

        # ✅ Load existing session data
        session_data = load_session(file_id)

        if not session_data:
            raise ValueError("Session data not found for given file ID")

        schema = {
            'file_id': file_id,
            'columns': list(df.columns),
            'row_count': len(df),
            'sample_data': df.head(10).replace([np.nan, np.inf, -np.inf], None).to_dict(orient='records')
        }

        schema = convert_nan_to_none(schema)

        # ✅ Update schema inside session
        session_data['schema'] = schema

        # ✅ Save updated session back
        save_session(file_id, session_data)

        # ✅ Save schema file separately too
        schema_file = os.path.join(UPLOAD_FOLDER, f"{file_id}_schema.json")
        os.makedirs(os.path.dirname(schema_file), exist_ok=True)

        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=4)

        print(f"✅ Schema saved successfully at: {schema_file}")

    except Exception as e:
        print(f"❌ Error generating schema: {str(e)}")
        traceback.print_exc()
        raise



def preprocess_data(data):
    """Preprocesses data (handling missing values and type conversion)."""
    try:
        for column in data.columns:
            if data[column].dtype == object:
                data[column] = data[column].fillna("Missing")
            else:
                data[column] = data[column].fillna(0)

            if data[column].dtype == object:
                numeric_data = pd.to_numeric(data[column], errors='coerce')
                if not numeric_data.isna().all():
                    data[column] = numeric_data

        data.replace([pd.NA, pd.NaT, float('nan'), float('inf'), float('-inf')], None, inplace=True)

        print(f"✅ Preprocessing completed: {data.head()}")
        return data

    except Exception as e:
        print(f"❌ Preprocessing Error: {str(e)}")
        traceback.print_exc()
        return data


def escape_sql_columns(query, columns):
    """
    Automatically escapes SQL column names with backticks to avoid syntax errors.
    """
    if not query or not columns:
        return query

    # ✅ Escape only valid column names
    for col in columns:
        # Escape column names with backticks if they contain underscores or special characters
        escaped_col = f"`{col}`" if "_" in col or col.lower() in ["date", "user", "group"] else col
        query = re.sub(rf'\b{re.escape(col)}\b', escaped_col, query, flags=re.IGNORECASE)

    return query


def escape_valid_columns(query, valid_columns):
    """
    Properly escapes only valid column names in the SQL query.
    - Ensures no double escaping issues.
    - Avoids modifying already escaped columns.
    """
    # ✅ Precompile regex for performance
    column_pattern = re.compile(r'`?(\w+)`?')

    def escape_column(match):
        col = match.group(1)
        if col in valid_columns:
            # ✅ Only add backticks if not already escaped
            return f"`{col}`" if not query.startswith("ALTER") else col
        return col

    # ✅ Apply proper escaping without double backticks
    escaped_query = column_pattern.sub(escape_column, query)

    print(f"🔥 Properly Escaped Query: {escaped_query}")
    return escaped_query




def ai(prompt, columns, query_language, file_id=None):
    """AI-powered query generation with strict SQL/Python validation, fallback, and Notebook logging."""

    # ✅ Format columns for AI prompt
    formatted_columns = ", ".join([f"`{col.lower()}`" for col in columns])

    # ✅ Language context
    if query_language == "sql":
        language_context = (
            "You are an SQL expert, Advanced Data Scientist. Write strictly valid MySQL queries.\n"
            "NEVER use invalid or non-existent columns. Use only valid columns."
            "Very strict warning don't explain anything."
            "Print the code only once and don't print any '''' unnecessarily."
        )
    else:
        language_context = (
            "You are a Python expert, Advanced Data Scientist. Write valid Pandas commands for DataFrame (df)."
            "NEVER use invalid or non-existent columns. Use only valid columns."
            "Very strict warning don't explain anything."
            "Print the code only once and don't print any '''' unnecessarily."
            "Don't write import statements."
        )

    # ✅ Improved prompt formatting
    formatted_prompt = (
        f"You are an expert, Advanced Data Scientist. Write strictly valid queries.\n"
        f"{language_context}\n"
        f"Columns: [{formatted_columns}].\n"
        f"User query: {prompt}.\n"
        f"Return the query type ('analysis' or 'modification' or 'visualization') on the first line, "
        f"followed by the valid SQL/Python query on the next line.\n"
        f"🚫 Only return the query type and query—no extra text."
    )

    try:
        print("🔥 Sending request to Together.AI...")

        # ✅ Together.AI API call
        response = together.Complete.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            prompt=formatted_prompt,
            max_tokens=256,
            temperature=0,
            top_p=0.5
        )

        # ✅ Extract AI response
        ai_response = response.get('output', {}).get('choices', [{}])[0].get('text', "").strip()
        print(f"🔥 Raw AI Response: {ai_response}")

        if not ai_response:
            print("❌ Empty AI response.")
            return "", ""

        # ✅ Parse the AI response
        match = re.search(r'^(analysis|modification|visualization)\n+(.+)', ai_response, re.IGNORECASE | re.DOTALL)

        if match:
            query_type = match.group(1).strip().lower()
            ai_query = match.group(2).strip()
        else:
            print("⚠️ Failed regex extraction. Attempting fallback extraction...")
            lines = ai_response.splitlines()
            query_type = lines[0].strip().lower() if len(lines) > 0 else ""
            ai_query = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        if query_type not in ['analysis', 'modification', 'visualization']:
            print("❌ Failed to extract valid query and code.")
            return "", ""

        print(f"✅ AI Query Type: {query_type}")
        print(f"✅ AI Query: {ai_query}")

        # ✅ If it's a Python query, append it into the notebook
        if query_language == "python" and file_id:
            notebook_path = f"/tmp/notebooks/{file_id}_session.ipynb"

            append_to_notebook(notebook_path, f"# 📋 Natural Language Query:\n# {prompt}")
            append_to_notebook(notebook_path, f"```python\n{ai_query}\n```")

            print(f"✅ Appended query to notebook: {notebook_path}")

        return query_type, ai_query

    except Exception as e:
        print(f"❌ AI Error: {str(e)}")
        traceback.print_exc()
        return "", ""



def spell_check_sql(query):
    """Spell-check SQL keywords and column names."""
    if not query:
        return ""

    spell = SpellChecker()

    # ✅ Split query into words, ignore None values
    words = query.split()

    # ✅ Identify misspelled words
    misspelled = spell.unknown(words)

    corrected_query = []
    for word in words:
        if word:
            corrected = spell.correction(word) if word in misspelled else word
            corrected_query.append(corrected if corrected else word)

    # ✅ Join corrected query
    return " ".join(corrected_query)


MAX_RETRIES = 3


@app.route('/query-sql/<file_id>', methods=['POST'])
def process_sql_query(file_id):
    """Handles SQL queries with MySQL, supporting multiline queries, validation, retry, and AI query debugging."""
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        # ✅ Load session
        session_data = load_session(file_id)

        if not session_data:
            return jsonify({'error': 'Invalid session ID'}), 400

        sql_details = session_data

        connection = pymysql.connect(
            host=sql_details['host'],
            user=sql_details['username'],
            password=sql_details['password'],
            database=sql_details['database'],
            port=sql_details['port'],
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False
        )

        cursor = connection.cursor()

        # ✅ Retry AI query generation
        retries = 0
        query_type, ai_query = "", ""

        while retries < MAX_RETRIES:
            query_type, ai_query = ai(query, pd.DataFrame(sql_details['original']).columns, "sql", file_id=file_id)

            if ai_query:
                break  # Exit retry loop if valid query is generated
            retries += 1
            print(f"🔁 Retry {retries}/{MAX_RETRIES}")

        if not ai_query:
            return jsonify({
                'error': 'AI failed to generate query after retries',
                'ai_query': ai_query
            }), 400

        # ✅ Handle multiple SQL statements
        results = []
        queries = ai_query.strip().split(";")
        queries = [q.strip() for q in queries if q.strip()]

        for q in queries:
            cursor.execute(q)
            if cursor.description:  # SELECT queries return results
                rows = cursor.fetchall()
                serialized_rows = serialize_data(rows)
                results.append({'query': q, 'result': serialized_rows})

        # ✅ Commit if modification
        if query_type == "modification":
            connection.commit()

            # ✅ Reload latest table
            reload_query = f"SELECT * FROM {sql_details['table']} LIMIT 100"
            cursor.execute(reload_query)
            updated_rows = cursor.fetchall()
            updated_serialized = serialize_data(updated_rows)

            # ✅ Update session
            df = pd.DataFrame(updated_serialized)
            session_data['original'] = df.to_dict()
            session_data['modified'] = df.to_dict()

            save_session(file_id, session_data)

        cursor.close()
        connection.close()

        return jsonify({
            'queries_executed': queries,
            'results': results,
            'ai_query': ai_query
        })

    except Exception as e:
        print(f"❌ General Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/reload/<file_id>', methods=['GET'])
def reload_table(file_id):
    """Fetches the latest table preview."""
    try:
        # ✅ Load session
        session_data = load_session(file_id)

        if not session_data:
            return jsonify({"error": "Invalid file ID"}), 400

        # ✅ Rebuild DataFrame from stored dict
        df = pd.DataFrame(session_data.get('modified', {}))

        if df.empty:
            return jsonify({"error": "No data available"}), 400

        # ✅ Return latest table preview
        preview = df.head(100).replace({np.nan: None}).to_dict(orient="records")

        return jsonify({
            "success": True,
            "preview": preview
        })

    except Exception as e:
        print(f"❌ Reload error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500



# --------- Flask Routes ---------
@app.route('/query/<file_id>', methods=['POST'])
def handle_query(file_id):
    """Handles both SQL and XLSX queries based on the data source."""
    data = request.get_json()
    query = data.get('query')
    query_language = data.get('query_language', 'python')
    data_source = data.get('data_source')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # ✅ Handle SQL separately
    if data_source == 'sql':
        return process_sql_query(file_id)

    try:
        # ✅ Load session
        session_data = load_session(file_id)

        if not session_data:
            return jsonify({'error': 'Invalid file ID'}), 400

        # ✅ Initialize modified dataset if missing
        if 'modified' not in session_data:
            session_data['modified'] = session_data['original']

        # ✅ Initialize history if missing
        if 'history' not in session_data:
            session_data['history'] = []

        df = pd.DataFrame(session_data['modified'])

        # ✅ AI-generated query and type
        query_type, ai_query = ai(query, df.columns, query_language, file_id=file_id)

        if not ai_query or not query_type:
            print("❌ Failed to generate a valid query.")
            return jsonify({
                "error": "Failed to generate a valid query.",
                "ai_query": "No AI Query",
                "translated_query": "No Translated Query"
            })

        # ✅ Clean the generated code
        cleaned_ai_query = "\n".join(
            line for line in ai_query.splitlines()
            if not (line.strip().startswith('import') or 'plt.show()' in line)
        )

        local_vars = {"df": df, "pd": pd, "plt": plt, "sns": sns, "np": np}
        result = None

        if query_type == "analysis":
            # ✅ Execute analysis
            result = eval(cleaned_ai_query, local_vars)

            def safe_serialize(obj):
                if isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, pd.DataFrame):
                    return obj.replace({np.nan: None}).to_dict(orient='records')
                elif isinstance(obj, (float, int, str)):
                    return obj
                else:
                    return str(obj)

            return jsonify({
                "query_type": "analysis",
                "ai_query": cleaned_ai_query,
                "translated_query": cleaned_ai_query,
                "result": safe_serialize(result)
            })

        elif query_type == "modification":
            # 🔁 Save current modified DataFrame into history
            session_data['history'].append(session_data['modified'])

            # ✅ Execute modification
            exec(cleaned_ai_query, local_vars)
            updated_df = local_vars['df']

            session_data['modified'] = updated_df.to_dict()
            save_session(file_id, session_data)

            preview_data = updated_df.head().replace({np.nan: None}).to_dict(orient="records")

            return jsonify({
                "query_type": "modification",
                "ai_query": cleaned_ai_query,
                "translated_query": cleaned_ai_query,
                "preview": preview_data
            })

        elif query_type == "visualization":
            os.makedirs('static', exist_ok=True)

            img_filename = f"{file_id}_viz_output.png"
            img_path = os.path.join('static', img_filename)

            plt.figure(figsize=(12, 6))
            exec(cleaned_ai_query, local_vars)

            if not plt.gcf().axes:
                raise ValueError("No plot was generated.")

            plt.savefig(img_path)
            plt.close()

            img_url = url_for('static', filename=img_filename) + f"?t={int(time.time())}"

            return jsonify({
                "query_type": "visualization",
                "ai_query": cleaned_ai_query,
                "translated_query": cleaned_ai_query,
                "visualization_output": img_url
            })

    except Exception as e:
        print(f"❌ Execution error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "error": f"❌ Execution error: {str(e)}",
            "ai_query": "No AI Query",
            "translated_query": "No Translated Query"
        })

# --------- Flask Routes ---------
@app.route('/save-schema/<file_id>', methods=['POST'])
def save_schema(file_id):
    """Saves schema descriptions with annotations."""
    try:
        # ✅ Load session
        session_data = load_session(file_id)

        if not session_data:
            return jsonify({'error': 'File ID not found'}), 404

        data = request.get_json()

        if not data or 'schema' not in data:
            return jsonify({'error': 'Invalid data format'}), 400

        schema = data.get('schema', [])
        query_language = data.get('query_language', 'SQL')

        # ✅ Save schema with annotations
        session_data['schema'] = {
            'annotations': schema,
            'query_language': query_language
        }

        # ✅ Save session back
        save_session(file_id, session_data)

        # ✅ Also save schema separately
        schema_file = os.path.join(UPLOAD_FOLDER, f"{file_id}_schema.json")
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(session_data['schema'], f, indent=4)

        print(f"✅ Schema saved successfully at {schema_file}")
        return jsonify({'message': 'Schema saved successfully'})

    except Exception as e:
        print(f"❌ Error saving schema: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500



@app.route('/get-data/<file_id>', methods=['GET'])
def get_data(file_id):
    """Serve the uploaded or modified dataset as JSON."""
    print(f"🔥 Fetching data for file_id: {file_id}")  # Debug log

    try:
        # ✅ Load session
        session_data = load_session(file_id)

        if not session_data:
            print("❌ File ID not found in session data.")
            return jsonify({"error": "File ID not found"}), 404

        # ✅ Use modified data if exists
        if 'modified' in session_data:
            df = pd.DataFrame(session_data['modified'])  # Modified dataset
            print("✅ Serving modified dataset.")
        else:
            file_path = session_data.get('file_path')

            if not file_path or not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return jsonify({"error": "File not found"}), 404

            # ✅ Load original file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                return jsonify({"error": "Unsupported file format"}), 400

            print(f"✅ Loaded original dataset: {file_path}")

            # ✅ Save original data into session
            session_data['original'] = df.to_dict()
            save_session(file_id, session_data)

        # 🔥 Convert NaN values to None
        preview_data = df.head(20).replace({np.nan: None}).to_dict(orient='records')

        return jsonify(preview_data)

    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Error loading dataset: {str(e)}"}), 500



@app.route('/get-data-null/<file_id>', methods=['GET'])
def get_data_with_null(file_id):
    """Serve the uploaded dataset as JSON with NaN converted to null."""
    print(f"🔥 Fetching data with NaN as null for file_id: {file_id}")  # Debug

    try:
        # ✅ Load session
        session_data = load_session(file_id)

        if not session_data:
            print("❌ File ID not found in session data.")
            return jsonify({"error": "File ID not found"}), 404

        file_path = session_data.get('file_path')

        if not file_path or not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return jsonify({"error": "File not found"}), 404

        # ✅ Load dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        print(f"✅ Dataset loaded successfully: {file_path}")

        # 🔥 Replace NaN with None
        df.replace({np.nan: None}, inplace=True)

        data = df.head(20).to_dict(orient='records')
        return jsonify(data)

    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Error loading dataset: {str(e)}"}), 500


@app.route('/cleaning/<file_id>', methods=['GET'])
def cleaning_page(file_id):
    """Renders the cleaning and visualization page for both MySQL and file-based datasets."""
    try:
        # ✅ Load session
        session_data = load_session(file_id)

        if not session_data:
            return "Error: File ID not found", 404

        # ✅ Handle MySQL datasets
        if session_data.get('db_type') == 'sql':
            print(f"🔥 Loading MySQL data for file_id: {file_id}")
            df = pd.DataFrame(session_data['original'])

        # ✅ Handle file-based datasets
        else:
            file_path = session_data.get('file_path')

            if not file_path or not os.path.exists(file_path):
                return "Error: File not found", 404

            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                return "Unsupported file format", 400

        sample_data = df.head(10).to_dict(orient='records')

        return render_template('cleaning.html', file_id=file_id, sample_data=sample_data)

    except Exception as e:
        print(f"❌ Error loading cleaning page: {str(e)}")
        traceback.print_exc()
        return f"Error loading cleaning page: {str(e)}", 500

@app.route('/')
def upload_page():
    """Renders the upload page."""
    return render_template('uploads.html')


@app.route('/describe/<file_id>', methods=['GET'])
def describe_file(file_id):
    """Handles file schema generation and error handling."""
    try:
        # ✅ Load session
        session_data = load_session(file_id)

        if not session_data:
            print(f"❌ File ID not found: {file_id}")
            return "Error: File ID not found", 404

        # ✅ Handle MySQL data
        if session_data.get('db_type') == 'sql':
            print(f"🔥 MySQL Data Loaded for file_id: {file_id}")
            df = pd.DataFrame(session_data['original'])

            if 'schema' not in session_data:
                generate_schema(file_id, df)
                session_data = load_session(file_id)  # Reload after schema generation

            schema = session_data.get('schema')

        # ✅ Handle File-based data
        else:
            file_path = session_data.get('file_path')

            if not file_path or not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return f"Error: File not found at {file_path}", 404

            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                return "Unsupported file format", 400

            # ✅ Generate schema
            generate_schema(file_id, df)
            session_data = load_session(file_id)  # Reload after schema generation
            schema = session_data.get('schema')

        return render_template('describe.html', schema=schema)

    except pd.errors.EmptyDataError:
        print(f"❌ Error: The dataset is empty.")
        return "Error: The dataset is empty.", 400

    except Exception as e:
        print(f"❌ Error loading schema: {str(e)}")
        traceback.print_exc()
        return f"Error loading schema: {str(e)}", 500



@app.route("/process-query/<file_id>", methods=["POST"])
def process_query(file_id):
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # ✅ Load session (optional here if you need file-specific info)
        session_data = load_session(file_id)

        if not session_data:
            return jsonify({"error": "Invalid file ID"}), 404

        # (Replace below with real AI logic if needed)
        processed_result = f"Processed query: {query} on file {file_id}"

        return jsonify({"query": query, "result": processed_result})

    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


@app.route('/get-schema/<file_id>', methods=['GET'])
def get_schema(file_id):
    """Fetches the schema data for a given file ID."""
    try:
        # ✅ Load session
        session_data = load_session(file_id)

        if not session_data:
            return jsonify({'error': 'File ID not found'}), 404

        schema = session_data.get('schema')

        if not schema:
            return jsonify({'error': 'Schema not found'}), 404

        # ✅ Convert NaN to None
        safe_schema = convert_nan_to_none(schema)

        return jsonify({'schema': safe_schema})

    except Exception as e:
        print(f"❌ Error fetching schema: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads with detailed error handling."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_id = str(uuid.uuid4())
        filename = sanitize_filename(file.filename)

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file_path = os.path.normpath(file_path)

        if len(file_path) > 260:
            raise OSError("File path too long! Please use a shorter filename.")

        # ✅ Save the file
        file.save(file_path)

        # ✅ Load dataset into Pandas
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # ✅ Create new session object
        session_data = {
            'file_path': file_path,
            'original': df.to_dict(),
            'modified': df.to_dict(),
            'history': []
        }

        # ✅ Save session
        save_session(file_id, session_data)

        print(f"✅ File uploaded successfully: {file_path}")

        return jsonify({'file_id': file_id})

    except OSError as e:
        print(f"❌ File path error: {str(e)}")
        return jsonify({'error': f"File path error: {str(e)}"}), 400

    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500


def get_model_instance(name, objective):
    if objective == 'classification':
        return {
            "RandomForestClassifier": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "XGBoostClassifier": XGBClassifier(eval_metric='logloss'),
            "LGBMClassifier": LGBMClassifier(),
            "CatBoostClassifier": CatBoostClassifier(verbose=0),
            "SVC": SVC(probability=True)
        }.get(name)

    elif objective == 'regression':
        return {
            "RandomForestRegressor": RandomForestRegressor(),
            "LinearRegression": LinearRegression(),
            "XGBoostRegressor": XGBRegressor(),
            "LGBMRegressor": LGBMRegressor(),
            "CatBoostRegressor": CatBoostRegressor(verbose=0),
            "SVR": SVR()
        }.get(name)

    elif objective == 'clustering':
        return {
            "KMeans": KMeans(n_clusters=3, random_state=42)
        }.get(name)

    return None


@app.route('/modeldev/<file_id>/columns', methods=['GET'])
def get_columns(file_id):
    try:
        session_data = load_session(file_id)

        if not session_data:
            return jsonify({"error": "File ID not found"}), 404

        df = pd.DataFrame(session_data["modified"])
        columns = list(df.columns)
        dtypes = df.dtypes.astype(str).to_dict()

        return jsonify({"columns": columns, "types": dtypes})

    except Exception as e:
        print(f"❌ Error fetching columns: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




@app.route('/modeldev/<file_id>/eda', methods=['POST'])
def generate_eda(file_id):
    try:
        session_data = load_session(file_id)

        if not session_data:
            return jsonify({"error": "File ID not found"}), 404

        df = pd.DataFrame(session_data['modified'])

        info = {
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'nulls': df.isnull().sum().to_dict(),
            'preview': df.head(5).replace({pd.NA: None}).to_dict(orient='records'),
            'summary': df.describe(include='all').replace({pd.NA: None}).to_dict()
        }
        return jsonify(info)

    except Exception as e:
        print(f"❌ Error generating EDA: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)})


@app.route('/modeldev/<file_id>/recommend', methods=['POST'])
def recommend_models(file_id):
    data = request.json
    objective = data.get("objective", "classification").lower()
    if objective == "classification":
        return jsonify({"models": ["RandomForestClassifier", "XGBoostClassifier", "LGBMClassifier", "CatBoostClassifier", "SVC"]})
    elif objective == "regression":
        return jsonify({"models": ["RandomForestRegressor", "XGBoostRegressor", "LGBMRegressor", "CatBoostRegressor", "LinearRegression"]})
    elif objective == "clustering":
        return jsonify({"models": ["KMeans"]})
    else:
        return jsonify({"error": "Unknown objective"}), 400


@app.route('/download-notebook/<file_id>', methods=['GET'])
def download_notebook(file_id):
    try:
        notebook_path = f"/tmp/notebooks/{file_id}_session.ipynb"

        if not os.path.exists(notebook_path):
            return jsonify({"error": "Notebook not found."}), 404

        return send_file(notebook_path, as_attachment=True)

    except Exception as e:
        print(f"❌ Error downloading notebook: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Error downloading notebook: {str(e)}"}), 500

# 🔥 Add this somewhere top globally
modelDescriptions = {
  "RandomForestClassifier": "🌲 Random Forest is an ensemble method that builds multiple decision trees and merges them to get more accurate and stable predictions. Excellent for handling tabular data with mixed types and missing values.",
  "LogisticRegression": "📈 A linear model used for binary classification. Fast, interpretable, and performs well when the data is linearly separable. Great for problems like spam detection, churn prediction, etc.",
  "XGBoostClassifier": "⚡ Extreme Gradient Boosting classifier. Highly optimized for performance and speed, often wins ML competitions. Handles missing values, regularization, and supports custom loss functions.",
  "LGBMClassifier": "🌿 LightGBM is a fast, efficient gradient boosting framework that uses histogram-based learning. Ideal for large datasets and high-dimensional features. Supports categorical features natively.",
  "CatBoostClassifier": "🐱 Gradient boosting by Yandex, designed for datasets with categorical variables. Requires minimal data preprocessing and tuning. Offers great out-of-the-box performance.",
  "SVC": "➗ Support Vector Classifier builds a decision boundary to separate classes with the maximum margin. Powerful for high-dimensional spaces but slower on large datasets.",
  "RandomForestRegressor": "🌳 Ensemble of decision trees for regression tasks. Reduces overfitting and handles non-linear relationships well. Suitable for structured/tabular data with moderate noise.",
  "LinearRegression": "📉 Basic linear approach for regression problems. Works well for linearly correlated data and is highly interpretable. Not suitable for complex patterns or outliers.",
  "XGBoostRegressor": "⚡ XGBoost for regression. Known for handling non-linear relationships with great accuracy. Includes regularization and missing value handling, making it robust and powerful.",
  "LGBMRegressor": "🚀 LightGBM for regression. Extremely fast and memory efficient. Performs well with large datasets, many features, and provides high accuracy with minimal tuning.",
  "CatBoostRegressor": "🐱 CatBoost for regression. Automatically deals with categorical features and works well with default parameters. Offers great generalization with minimal overfitting.",
  "SVR": "🧮 Support Vector Regression. Tries to fit the best line within a margin of tolerance. Works well for small- to medium-sized datasets, but sensitive to parameter tuning and scaling.",
  "KMeans": "🎯 Unsupervised clustering algorithm. Divides data into K distinct non-overlapping groups based on feature similarity. Best for numerical data where grouping is meaningful."
}

# 🔥 Now modified route
@app.route('/modeldev/<file_id>/train', methods=['POST'])
def train_model(file_id):
    try:
        session_data = load_session(file_id)

        if not session_data:
            return jsonify({"error": "File ID not found"}), 404

        df = pd.DataFrame(session_data["modified"])
        data = request.get_json()

        objective = data['objective'].lower()
        model_name = data['model']
        target_col = data.get('target')
        selected_features = data.get('features', [])
        train_split = data['train_split'] / 100.0
        cv = int(data.get("cv", 5))

        model = get_model_instance(model_name, objective)
        if not model:
            return jsonify({"error": f"Model {model_name} not supported"}), 400

        if objective != 'clustering' and target_col not in df.columns:
            return jsonify({"error": f"Target column '{target_col}' not found in dataset"}), 400

        if objective == 'clustering':
            X = df.select_dtypes(include=['number']).dropna()
            model.fit(X)
            y_pred = model.labels_
            metrics = {'clusters': y_pred.tolist()}
            scores = None

        else:
            if not selected_features or not all(col in df.columns for col in selected_features):
                return jsonify({"error": "Selected features are invalid or missing"}), 400

            X = df[selected_features].dropna()
            y = df[target_col].loc[X.index]

            X = pd.get_dummies(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1 - train_split, random_state=42)

            model.fit(X_train, y_train)
            scores = cross_val_score(model, X_train, y_train, cv=cv)

            y_pred = model.predict(X_test)
            metrics = {}

            if objective == "classification":
                metrics['report'] = classification_report(y_test, y_pred, output_dict=True)
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                try:
                    metrics['roc_auc'] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                except:
                    metrics['roc_auc'] = None
            elif objective == "regression":
                metrics['mse'] = mean_squared_error(y_test, y_pred)

        # Save model
        filename = f"{file_id}_{model_name}.pkl"
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump(model, path)

        # 🔥 Fetch model description if available
        model_description = modelDescriptions.get(model_name, "ℹ️ No description available.")

        return jsonify({
            "success": True,
            "cv_score": scores.mean() if scores is not None else None,
            "metrics": metrics,
            "model_path": filename,
            "model": model_name,
            "model_description": model_description  # 🚀 Added!
        })

    except Exception as e:
        print(f"❌ Error training model: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# --------- Run App ---------
if __name__ == '__main__':
    app.run(debug=True)

