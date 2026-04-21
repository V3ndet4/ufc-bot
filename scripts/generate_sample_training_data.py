"""
Generate Sample Training Data

Creates synthetic historical data for testing the model pipeline.
Real data would come from scraping ufcstats.com and bestfightodds.com.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def generate_sample_database(db_path: str = "data/historical_ufc.db", n_fights: int = 1000):
    """Generate a sample database for testing."""
    
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop existing tables
    cursor.execute("DROP TABLE IF EXISTS fights")
    cursor.execute("DROP TABLE IF EXISTS events")
    cursor.execute("DROP TABLE IF EXISTS fight_odds")
    cursor.execute("DROP TABLE IF EXISTS fighter_histories")
    
    # Create tables
    cursor.execute("""
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            event_name TEXT NOT NULL,
            event_date TEXT NOT NULL,
            location TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE fights (
            fight_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            fighter_a TEXT NOT NULL,
            fighter_b TEXT NOT NULL,
            winner TEXT,
            result_method TEXT,
            round INTEGER,
            time TEXT,
            weight_class TEXT,
            went_decision INTEGER,
            ended_inside_distance INTEGER
        )
    """)
    
    cursor.execute("""
        CREATE TABLE fight_odds (
            odds_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fight_id TEXT NOT NULL,
            bookmaker TEXT,
            open_a_odds INTEGER,
            open_b_odds INTEGER,
            close_a_odds INTEGER,
            close_b_odds INTEGER
        )
    """)
    
    cursor.execute("""
        CREATE TABLE fighter_histories (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fighter_name TEXT NOT NULL,
            fight_id TEXT NOT NULL,
            event_date TEXT NOT NULL,
            height_in REAL,
            reach_in REAL,
            ufc_wins INTEGER,
            ufc_losses INTEGER,
            total_ufc_fights INTEGER,
            sig_strikes_landed_per_min REAL,
            sig_strikes_absorbed_per_min REAL,
            takedown_avg REAL,
            finish_rate REAL
        )
    """)
    
    # Generate fighters
    n_fighters = 200
    fighters = []
    
    first_names = ["Jon", "Conor", "Khabib", "Israel", "Alex", "Max", "Dustin", "Charles",
                   "Amanda", "Valentina", "Weili", "Rose", "Joanna", "Holly", "Miesha", "Ronda",
                   "Stipe", "Francis", "Derrick", "Curtis", "Ciryl", "Tom", "Sergei", "Alexander",
                   "Robert", "Marvin", "Paulo", "Jared", "Sean", "Petr", "Merab", "Henry"]
    last_names = ["Jones", "McGregor", "Nurmagomedov", "Adesanya", "Pereira", "Holloway", 
                  "Poirier", "Oliveira", "Nunes", "Shevchenko", "Zhang", "Namajunas",
                  "Jedrzejczyk", "Holm", "Tate", "Rousey", "Miocic", "Ngannou", "Lewis",
                  "Blaydes", "Gane", "Aspinall", "Pavlovich", "Volkanovski", "Whittaker",
                  "Vettori", "Costa", "Cannonier", "O'Malley", "Yan", "Dvalishvili", "Cejudo"]
    
    for i in range(n_fighters):
        name = f"{random.choice(first_names)} {random.choice(last_names)} {i}"
        height = random.randint(64, 76)  # 5'4" to 6'4"
        reach = height + random.randint(-2, 6)
        
        fighters.append({
            "name": name,
            "height": height,
            "reach": reach,
            "base_striking": random.uniform(2.0, 6.0),
            "base_grappling": random.uniform(0.5, 3.0),
            "base_finish": random.uniform(0.1, 0.5),
        })
    
    # Generate fights over time
    start_date = datetime(2020, 1, 1)
    
    for fight_idx in range(n_fights):
        # Random event
        event_date = start_date + timedelta(days=random.randint(0, 2000))
        event_id = f"event_{fight_idx // 10}"
        event_name = f"UFC Fight Night {fight_idx // 10}"
        
        # Only insert event once
        if fight_idx % 10 == 0:
            cursor.execute("""
                INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?)
            """, (event_id, event_name, event_date.strftime("%Y-%m-%d"), "Las Vegas, NV"))
        
        # Pick two random fighters
        f_a = random.choice(fighters)
        f_b = random.choice([f for f in fighters if f != f_a])
        
        # Simulate fight based on skill
        skill_diff = (
            (f_a["base_striking"] - f_b["base_striking"]) * 0.4 +
            (f_a["base_grappling"] - f_b["base_grappling"]) * 0.3 +
            (f_a["reach"] - f_b["reach"]) * 0.05 +
            random.gauss(0, 0.5)  # Randomness
        )
        
        # Market odds (slightly noisy around true probability)
        true_prob_a = 1 / (1 + 2.718 ** (-skill_diff))
        market_prob_a = true_prob_a + random.gauss(0, 0.05)
        market_prob_a = max(0.15, min(0.85, market_prob_a))
        
        # Convert to American odds
        if market_prob_a > 0.5:
            odds_a = int(-100 * market_prob_a / (1 - market_prob_a))
        else:
            odds_a = int(100 * (1 - market_prob_a) / market_prob_a)
        
        market_prob_b = 1 - market_prob_a
        if market_prob_b > 0.5:
            odds_b = int(-100 * market_prob_b / (1 - market_prob_b))
        else:
            odds_b = int(100 * (1 - market_prob_b) / market_prob_b)
        
        # Determine winner
        if random.random() < true_prob_a:
            winner = f_a["name"]
            result = "Decision" if random.random() > 0.5 else "KO/TKO"
        else:
            winner = f_b["name"]
            result = "Decision" if random.random() > 0.5 else "KO/TKO"
        
        went_decision = 1 if result == "Decision" else 0
        ended_inside = 1 if result in ["KO/TKO", "Submission"] else 0
        
        fight_id = f"{event_id}_{fight_idx}"
        
        # Insert fight (11 columns: no fight_id primary key in this schema)
        cursor.execute("""
            INSERT INTO fights VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fight_id, event_id, f_a["name"], f_b["name"], winner, result,
            random.randint(1, 5), "5:00", "Middleweight", went_decision, ended_inside
        ))
        
        # Insert odds
        cursor.execute("""
            INSERT INTO fight_odds (fight_id, bookmaker, open_a_odds, open_b_odds, close_a_odds, close_b_odds)
            VALUES (?, 'synthetic', ?, ?, ?, ?)
        """, (fight_id, odds_a + random.randint(-20, 20), odds_b + random.randint(-20, 20), odds_a, odds_b))
        
        # Insert fighter histories (simplified - cumulative stats before this fight)
        for f, is_a in [(f_a, True), (f_b, False)]:
            # Randomize their career stats slightly
            wins = random.randint(3, 15)
            losses = random.randint(0, 5)
            
            cursor.execute("""
                INSERT INTO fighter_histories 
                (fighter_name, fight_id, event_date, height_in, reach_in, ufc_wins, ufc_losses,
                 total_ufc_fights, sig_strikes_landed_per_min, sig_strikes_absorbed_per_min,
                 takedown_avg, finish_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f["name"], fight_id, event_date.strftime("%Y-%m-%d"),
                f["height"], f["reach"], wins, losses, wins + losses,
                f["base_striking"], 4.0 + random.gauss(0, 0.5),
                f["base_grappling"], f["base_finish"]
            ))
    
    conn.commit()
    conn.close()
    
    print(f"Generated {n_fights} sample fights in {db_path}")
    print(f"Fighter records, odds, and historical stats included")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample training data")
    parser.add_argument("--db-path", default="data/historical_ufc.db")
    parser.add_argument("--n-fights", type=int, default=1000)
    
    args = parser.parse_args()
    
    generate_sample_database(args.db_path, args.n_fights)


if __name__ == "__main__":
    main()
