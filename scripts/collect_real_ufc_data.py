"""
Collect Real UFC Historical Data

Downloads actual UFC fight data from ufcstats.com.
This replaces the synthetic data with real fights for model training.

Usage:
    python scripts/collect_real_ufc_data.py --years 2020 2021 2022 2023 2024
    
This will scrape:
    - All events in specified years
    - All fights from each event
    - Basic fighter profiles
    
Takes 30-60 minutes to run for 5 years of data.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_sources.ufc_stats_scraper import UFCStatsScraper


def init_database(db_path: str) -> None:
    """Create database with proper schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.executescript("""
        DROP TABLE IF EXISTS real_fights;
        DROP TABLE IF EXISTS real_events;
        DROP TABLE IF EXISTS real_fighters;
        
        CREATE TABLE real_events (
            event_id TEXT PRIMARY KEY,
            event_name TEXT NOT NULL,
            event_date TEXT,
            event_location TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE real_fights (
            fight_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            event_date TEXT,
            fighter_a TEXT NOT NULL,
            fighter_b TEXT NOT NULL,
            winner TEXT,
            result_method TEXT,
            round INTEGER,
            time TEXT,
            weight_class TEXT,
            went_decision BOOLEAN,
            ended_inside_distance BOOLEAN,
            FOREIGN KEY (event_id) REFERENCES real_events(event_id)
        );
        
        CREATE TABLE real_fighters (
            fighter_id TEXT PRIMARY KEY,
            fighter_name TEXT NOT NULL,
            height_in REAL,
            weight_lbs REAL,
            reach_in REAL,
            stance TEXT,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            draws INTEGER DEFAULT 0,
            profile_url TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_fights_event ON real_fights(event_id);
        CREATE INDEX idx_fights_fighter_a ON real_fights(fighter_a);
        CREATE INDEX idx_fights_fighter_b ON real_fights(fighter_b);
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized: {db_path}")


def collect_events(scraper: UFCStatsScraper, target_years: list[int], db_path: str) -> list[dict]:
    """Collect events for specified years."""
    all_events = scraper.get_event_list()
    
    # Filter by year
    filtered_events = []
    for event in all_events:
        try:
            event_date = datetime.strptime(event['date'], '%B %d, %Y')
            if event_date.year in target_years:
                filtered_events.append(event)
        except:
            continue
    
    print(f"\nFound {len(filtered_events)} events in years {target_years}")
    
    # Save events to database
    conn = sqlite3.connect(db_path)
    for event in filtered_events:
        conn.execute("""
            INSERT OR REPLACE INTO real_events (event_id, event_name, event_date)
            VALUES (?, ?, ?)
        """, (event['id'], event['name'], event['date']))
    conn.commit()
    conn.close()
    
    return filtered_events


def collect_fights(scraper: UFCStatsScraper, events: list[dict], db_path: str) -> dict:
    """Collect all fights from events."""
    conn = sqlite3.connect(db_path)
    
    all_fights = []
    unique_fighters = set()
    
    for i, event in enumerate(events):
        print(f"\n[{i+1}/{len(events)}] Collecting fights from: {event['name']}")
        
        try:
            fights = scraper.get_event_fights(event['url'])
            
            for fight in fights:
                # Generate fight ID
                fight_id = f"{event['id']}_{fight['fighter_a'].replace(' ', '_')}_{fight['fighter_b'].replace(' ', '_')}"
                
                # Parse method for decision/inside distance flags
                method = fight['method']
                went_decision = 1 if 'Decision' in method else 0
                ended_inside = 1 if any(x in method for x in ['KO', 'TKO', 'Submission']) else 0
                
                # Determine actual winner
                winner = fight['winner']
                if winner == 'DRAW':
                    winner = None  # Exclude draws from training
                elif winner == 'NC':
                    winner = None  # Exclude NCs
                
                conn.execute("""
                    INSERT OR REPLACE INTO real_fights 
                    (fight_id, event_id, event_date, fighter_a, fighter_b, winner,
                     result_method, round, time, weight_class, went_decision, ended_inside_distance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fight_id, event['id'], event['date'], 
                    fight['fighter_a'], fight['fighter_b'], winner,
                    method, fight.get('round', 0), fight.get('time', ''),
                    'Unknown', went_decision, ended_inside
                ))
                
                all_fights.append(fight)
                unique_fighters.add(fight['fighter_a'])
                unique_fighters.add(fight['fighter_b'])
            
            print(f"  ✓ Saved {len(fights)} fights")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    print(f"\n{'='*50}")
    print(f"Total fights collected: {len(all_fights)}")
    print(f"Unique fighters: {len(unique_fighters)}")
    
    return {
        'n_fights': len(all_fights),
        'n_fighters': len(unique_fighters),
        'fighters': list(unique_fighters)
    }


def main():
    parser = argparse.ArgumentParser(description="Collect real UFC historical data")
    parser.add_argument("--years", nargs="+", type=int, 
                       default=[2020, 2021, 2022, 2023, 2024],
                       help="Years to collect")
    parser.add_argument("--db-path", default="data/historical_ufc_real.db")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Seconds between requests")
    parser.add_argument("--limit-events", type=int, default=None,
                       help="Limit to N events (for testing)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("UFC HISTORICAL DATA COLLECTOR")
    print("="*60)
    print(f"Years: {args.years}")
    print(f"Database: {args.db_path}")
    print(f"Delay: {args.delay}s")
    print()
    
    # Initialize database
    init_database(args.db_path)
    
    # Create scraper
    scraper = UFCStatsScraper(delay=args.delay)
    
    # Collect events
    events = collect_events(scraper, args.years, args.db_path)
    
    if args.limit_events:
        events = events[:args.limit_events]
        print(f"Limited to {len(events)} events for testing")
    
    # Collect fights
    results = collect_fights(scraper, events, args.db_path)
    
    # Save fighter list for next step
    fighters_path = Path(args.db_path).parent / "fighters_to_scrape.json"
    with open(fighters_path, 'w') as f:
        json.dump(results['fighters'], f, indent=2)
    
    print(f"\n{'='*60}")
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"Database: {args.db_path}")
    print(f"Fights: {results['n_fights']}")
    print(f"Fighters: {results['n_fighters']}")
    print()
    print("Next steps:")
    print("1. Collect fighter profiles:")
    print(f"   python scripts/collect_fighter_profiles.py --db-path {args.db_path}")
    print("2. Collect historical odds:")
    print("   python scripts/collect_historical_odds.py --db-path {args.db_path}")
    print("3. Train model:")
    print("   python models/trainer.py --db-path {args.db_path}")


if __name__ == "__main__":
    main()
