#!/usr/bin/env python3
"""
Create SQLite database for Optuna dashboard with both 14res and 15res results
"""

import sqlite3
import json
import os
from pathlib import Path

def create_optuna_database():
    """Create SQLite database compatible with Optuna dashboard"""
    
    db_path = "optuna_results/hyperparameter_dashboard.db"
    
    # Create database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create studies table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS studies (
            study_id INTEGER PRIMARY KEY,
            study_name TEXT UNIQUE,
            direction TEXT,
            user_attrs TEXT
        )
    ''')
    
    # Create trials table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trials (
            trial_id INTEGER PRIMARY KEY,
            study_id INTEGER,
            number INTEGER,
            value REAL,
            datetime_start TEXT,
            datetime_complete TEXT,
            state TEXT,
            params TEXT,
            user_attrs TEXT,
            FOREIGN KEY (study_id) REFERENCES studies (study_id)
        )
    ''')
    
    # Insert 14res study
    cursor.execute('''
        INSERT OR REPLACE INTO studies (study_id, study_name, direction, user_attrs)
        VALUES (1, 'd2e2s_14res_balanced_1759896139', 'maximize', '{"dataset": "14res"}')
    ''')
    
    # Insert 15res study  
    cursor.execute('''
        INSERT OR REPLACE INTO studies (study_id, study_name, direction, user_attrs)
        VALUES (2, 'd2e2s_15res_balanced_1760675899', 'maximize', '{"dataset": "15res"}')
    ''')
    
    # Load and insert 14res trials
    try:
        with open('optuna_results/d2e2s_14res_balanced_1759896139_all_trials.json', 'r') as f:
            trials_14res = json.load(f)
        
        for trial in trials_14res:
            cursor.execute('''
                INSERT OR REPLACE INTO trials 
                (trial_id, study_id, number, value, state, params, user_attrs)
                VALUES (?, 1, ?, ?, ?, ?, ?)
            ''', (
                trial['number'] + 1,  # trial_id
                trial['number'],      # number
                trial['value'],       # value
                trial['state'],       # state
                json.dumps(trial['params']),  # params
                '{"dataset": "14res"}'        # user_attrs
            ))
        print(f"✅ Inserted {len(trials_14res)} trials for 14res")
    except FileNotFoundError:
        print("⚠️  14res trials file not found")
    
    # Load and insert 15res trials
    try:
        with open('optuna_results/d2e2s_15res_balanced_1760675899_all_trials.json', 'r') as f:
            trials_15res = json.load(f)
        
        for trial in trials_15res:
            cursor.execute('''
                INSERT OR REPLACE INTO trials 
                (trial_id, study_id, number, value, state, params, user_attrs)
                VALUES (?, 2, ?, ?, ?, ?, ?)
            ''', (
                trial['number'] + 100,  # trial_id (offset to avoid conflicts)
                trial['number'],        # number
                trial['value'],         # value
                trial['state'],         # state
                json.dumps(trial['params']),  # params
                '{"dataset": "15res"}'        # user_attrs
            ))
        print(f"✅ Inserted {len(trials_15res)} trials for 15res")
    except FileNotFoundError:
        print("⚠️  15res trials file not found")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database created: {db_path}")
    return db_path

def main():
    print("🚀 Creating Optuna dashboard database...")
    db_path = create_optuna_database()
    
    print("\n🌐 To start the dashboard, run:")
    print(f"export PATH=\"$PATH:/Users/vishal.thenuwara/.local/bin\"")
    print(f"optuna-dashboard sqlite:///{db_path}")
    print("\n📊 Then open: http://localhost:8080")

if __name__ == "__main__":
    main()