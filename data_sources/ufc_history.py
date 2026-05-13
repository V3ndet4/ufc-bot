"""
UFC Historical Data Collector

Scrapes historical UFC fight data from ufcstats.com
Builds a labeled dataset for model training.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.ufcstats.com"
EVENT_LIST_URL = f"{BASE_URL}/statistics/events/completed?page=all"


@dataclass
class FightResult:
    """Single fight outcome with metadata."""
    event_id: str
    event_name: str
    event_date: str
    fighter_a: str
    fighter_b: str
    winner: Optional[str]  # None for draws/NCs
    result_method: str
    round: int
    time: str
    weight_class: str
    is_title_fight: bool


@dataclass 
class FightOdds:
    """Odds for a specific fight."""
    event_id: str
    fighter_a: str
    fighter_b: str
    bookmaker: str
    open_a_odds: Optional[int]
    open_b_odds: Optional[int]
    close_a_odds: Optional[int]
    close_b_odds: Optional[int]


class UFCHistoricalCollector:
    """Collects historical UFC data from ufcstats.com"""
    
    def __init__(self, db_path: str = "data/historical_ufc.db", delay: float = 1.0):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        self._init_db()
    
    def _init_db(self) -> None:
        """Create database tables if not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_name TEXT NOT NULL,
                event_date TEXT NOT NULL,
                location TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Fights table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fights (
                fight_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                fighter_a TEXT NOT NULL,
                fighter_b TEXT NOT NULL,
                winner TEXT,
                result_method TEXT,
                round INTEGER,
                time TEXT,
                weight_class TEXT,
                is_title_fight BOOLEAN DEFAULT 0,
                went_decision BOOLEAN,
                ended_inside_distance BOOLEAN,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            )
        """)
        
        # Fighter history (stats as of fight time)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fighter_histories (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fighter_name TEXT NOT NULL,
                fight_id TEXT NOT NULL,
                event_date TEXT NOT NULL,
                ufc_wins INTEGER DEFAULT 0,
                ufc_losses INTEGER DEFAULT 0,
                ufc_draws INTEGER DEFAULT 0,
                total_ufc_fights INTEGER DEFAULT 0,
                height_in REAL,
                reach_in REAL,
                stance TEXT,
                FOREIGN KEY (fight_id) REFERENCES fights(fight_id)
            )
        """)
        
        # Odds table (to be filled from other sources)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fight_odds (
                odds_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fight_id TEXT NOT NULL,
                bookmaker TEXT,
                open_a_odds INTEGER,
                open_b_odds INTEGER,
                close_a_odds INTEGER,
                close_b_odds INTEGER,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (fight_id) REFERENCES fights(fight_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def fetch_event_list(self) -> list[dict]:
        """Fetch all completed UFC events."""
        print(f"Fetching event list from {EVENT_LIST_URL}")
        response = self.session.get(EVENT_LIST_URL)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        events = []
        
        # Event rows have class "b-statistics__table-row"
        for row in soup.select("tr.b-statistics__table-row")[1:]:  # Skip header
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
                
            link = cols[0].find("a", href=True)
            if not link:
                continue
                
            event_id = self._extract_event_id(link["href"])
            event_name = link.text.strip()
            event_date = cols[0].find("span", class_="b-statistics__date") 
            event_date = event_date.text.strip() if event_date else ""
            
            events.append({
                "event_id": event_id,
                "event_name": event_name,
                "event_date": event_date,
                "event_url": link["href"]
            })
        
        print(f"Found {len(events)} events")
        return events
    
    def fetch_event_details(self, event_url: str) -> list[FightResult]:
        """Fetch all fights from a specific event."""
        print(f"Fetching event: {event_url}")
        response = self.session.get(event_url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        fights = []
        
        # Extract event info
        event_id = self._extract_event_id(event_url)
        event_name_elem = soup.select_one("h2.b-content__title")
        event_name = event_name_elem.text.strip() if event_name_elem else "Unknown"
        
        event_date_elem = soup.select_one("li.b-list__box-list-item:contains('Date:')")
        event_date = ""
        if event_date_elem:
            date_text = event_date_elem.text.replace("Date:", "").strip()
            try:
                event_date = datetime.strptime(date_text, "%B %d, %Y").strftime("%Y-%m-%d")
            except ValueError:
                event_date = date_text
        
        # Fight rows
        fight_rows = soup.select("tbody tr.b-fight-details__table-row")
        
        for row in fight_rows:
            try:
                fight = self._parse_fight_row(row, event_id, event_name, event_date)
                if fight:
                    fights.append(fight)
            except Exception as e:
                print(f"Error parsing fight row: {e}")
                continue
        
        time.sleep(self.delay)
        return fights
    
    def _parse_fight_row(self, row, event_id: str, event_name: str, event_date: str) -> Optional[FightResult]:
        """Parse a single fight row."""
        cols = row.find_all("td")
        if len(cols) < 5:
            return None
        
        # Fighter names
        fighter_links = cols[1].find_all("p", class_="b-fight-details__table-text")
        if len(fighter_links) < 2:
            return None
            
        fighter_a = fighter_links[0].text.strip()
        fighter_b = fighter_links[1].text.strip()
        
        # Result
        result_cell = cols[0].find("p", class_="b-fight-details__table-text")
        result = result_cell.text.strip() if result_cell else ""
        
        # Determine winner
        winner = None
        if "win" in result.lower():
            # Check which fighter won (first p is usually winner if win is shown)
            winner = fighter_a  # Simplified - need to check actual logic
        elif "draw" in result.lower() or "nc" in result.lower():
            winner = None
        
        # Method
        method = self._table_cell_text(cols[-3]) or "Unknown"

        # Round and time
        round_num = int(self._table_cell_text(cols[-2]) or 0)

        time_str = self._table_cell_text(cols[-1])
        
        # Weight class (from bout info)
        weight_class = "Unknown"
        is_title = False
        
        return FightResult(
            event_id=event_id,
            event_name=event_name,
            event_date=event_date,
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            winner=winner,
            result_method=method,
            round=round_num,
            time=time_str,
            weight_class=weight_class,
            is_title_fight=is_title
        )

    def _table_cell_text(self, cell) -> str:
        texts = [" ".join(text.split()) for text in cell.stripped_strings if str(text).strip()]
        cleaned = " ".join(texts)
        cleaned = re.sub(r"\bImage\b", "", cleaned, flags=re.IGNORECASE)
        return " ".join(cleaned.split()).strip()
    
    def _extract_event_id(self, url: str) -> str:
        """Extract event ID from URL."""
        match = re.search(r"/event-details/(\w+)", url)
        return match.group(1) if match else url.split("/")[-1]
    
    def save_fights(self, fights: list[FightResult]) -> None:
        """Save fights to database."""
        conn = sqlite3.connect(self.db_path)
        
        for fight in fights:
            fight_id = f"{fight.event_id}_{fight.fighter_a}_{fight.fighter_b}".replace(" ", "_")
            
            # Determine result flags
            went_decision = fight.result_method == "Decision"
            ended_inside = fight.result_method in ["KO/TKO", "Submission", "Technical Submission"]
            
            conn.execute("""
                INSERT OR REPLACE INTO fights 
                (fight_id, event_id, fighter_a, fighter_b, winner, result_method, 
                 round, time, weight_class, is_title_fight, went_decision, ended_inside_distance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fight_id, fight.event_id, fight.fighter_a, fight.fighter_b,
                fight.winner, fight.result_method, fight.round, fight.time,
                fight.weight_class, fight.is_title_fight, went_decision, ended_inside
            ))
        
        conn.commit()
        conn.close()
    
    def run_full_collection(self, limit: Optional[int] = None) -> None:
        """Collect all historical fight data."""
        events = self.fetch_event_list()
        
        if limit:
            events = events[:limit]
        
        total_fights = 0
        for i, event in enumerate(events):
            print(f"\n[{i+1}/{len(events)}] Processing: {event['event_name']}")
            
            try:
                fights = self.fetch_event_details(event["event_url"])
                self.save_fights(fights)
                total_fights += len(fights)
                print(f"  Saved {len(fights)} fights")
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
        
        print(f"\nCollection complete! Total fights saved: {total_fights}")


def main():
    """Run historical data collection."""
    collector = UFCHistoricalCollector()
    
    # Start with last 100 events for testing, then expand
    collector.run_full_collection(limit=100)


if __name__ == "__main__":
    main()
