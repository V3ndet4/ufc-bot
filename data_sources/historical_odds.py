"""
Historical Odds Collector

Collects historical closing lines from BestFightOdds.
This is critical for model training — we need the odds *at fight time*.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.bestfightodds.com"


@dataclass
class HistoricalOdds:
    """Historical odds for a single fight."""
    event_name: str
    event_date: str
    fighter_a: str
    fighter_b: str
    open_a_odds: Optional[int]
    open_b_odds: Optional[int]
    close_a_odds: Optional[int]  # Best closing line
    close_b_odds: Optional[int]
    consensus_a: Optional[int]  # Consensus closing
    consensus_b: Optional[int]


class BestFightOddsHistorical:
    """Scrape historical odds from BestFightOdds."""
    
    def __init__(self, db_path: str = "data/historical_ufc.db", delay: float = 1.5):
        self.db_path = Path(db_path)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def fetch_event_odds(self, event_url: str) -> list[HistoricalOdds]:
        """
        Fetch odds for a specific event.
        BestFightOdds shows open/close for each bookmaker.
        """
        print(f"Fetching odds: {event_url}")
        response = self.session.get(event_url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        odds_list = []
        
        # Event title
        title_elem = soup.select_one("h1")
        event_name = title_elem.text.strip() if title_elem else "Unknown"
        
        # Event date
        date_elem = soup.select_one("span.date")
        event_date = ""
        if date_elem:
            date_text = date_elem.text.strip()
            # Parse "Mar 21, 2026" format
            try:
                from datetime import datetime
                event_date = datetime.strptime(date_text, "%b %d, %Y").strftime("%Y-%m-%d")
            except:
                event_date = date_text
        
        # Fight tables - each fight is a table row
        fight_rows = soup.select("table tbody tr")
        
        for row in fight_rows:
            try:
                fight_odds = self._parse_odds_row(row, event_name, event_date)
                if fight_odds:
                    odds_list.append(fight_odds)
            except Exception as e:
                print(f"Error parsing row: {e}")
                continue
        
        time.sleep(self.delay)
        return odds_list
    
    def _parse_odds_row(self, row, event_name: str, event_date: str) -> Optional[HistoricalOdds]:
        """Parse a single odds row."""
        # Fighter names
        fighter_links = row.select("a.fighter-name")
        if len(fighter_links) < 2:
            return None
        
        fighter_a = fighter_links[0].text.strip()
        fighter_b = fighter_links[1].text.strip()
        
        # Odds cells - look for open/close
        odds_cells = row.select("td")
        if len(odds_cells) < 3:
            return None
        
        # BestFightOdds structure varies
        # Usually shows multiple bookmakers with open/close
        # We'll grab the "best" closing odds (most favorable to bettor)
        
        open_a, open_b = None, None
        close_a, close_b = None, None
        
        # Try to extract from data attributes or cells
        for cell in odds_cells:
            cell_text = cell.get_text(strip=True)
            # Look for odds patterns like -150, +200
            odds_matches = re.findall(r'([+-]?\d+)', cell_text)
            if len(odds_matches) >= 2:
                # This is simplified - real scraping needs more precision
                try:
                    close_a = int(odds_matches[0])
                    close_b = int(odds_matches[1])
                except:
                    pass
        
        return HistoricalOdds(
            event_name=event_name,
            event_date=event_date,
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            open_a_odds=open_a,
            open_b_odds=open_b,
            close_a_odds=close_a,
            close_b_odds=close_b,
            consensus_a=close_a,
            consensus_b=close_b
        )
    
    def search_events(self, fighter_name: Optional[str] = None, 
                     year: Optional[int] = None) -> list[dict]:
        """
        Search for events by fighter or year.
        Returns list of event URLs to scrape.
        """
        # BestFightOdds doesn't have a great search, so we'll use their event list
        # This requires scraping their events page
        
        events = []
        
        # Try to get events by year
        if year:
            url = f"{BASE_URL}/events/{year}"
            try:
                response = self.session.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                
                event_links = soup.select("a[href*='/events/']")
                for link in event_links:
                    if "/events/" in link.get("href", ""):
                        events.append({
                            "event_name": link.text.strip(),
                            "event_url": urljoin(BASE_URL, link["href"]),
                            "year": year
                        })
            except Exception as e:
                print(f"Error fetching {url}: {e}")
        
        return events
    
    def save_odds_to_db(self, odds_list: list[HistoricalOdds]) -> None:
        """Save odds to database, matching against existing fights."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for odds in odds_list:
            # Find matching fight
            cursor.execute("""
                SELECT fight_id FROM fights 
                WHERE (fighter_a = ? AND fighter_b = ?)
                   OR (fighter_a = ? AND fighter_b = ?)
                ORDER BY event_date DESC
                LIMIT 1
            """, (odds.fighter_a, odds.fighter_b, odds.fighter_b, odds.fighter_a))
            
            result = cursor.fetchone()
            if result:
                fight_id = result[0]
                
                cursor.execute("""
                    INSERT OR REPLACE INTO fight_odds
                    (fight_id, bookmaker, open_a_odds, open_b_odds, 
                     close_a_odds, close_b_odds)
                    VALUES (?, 'bestfightodds', ?, ?, ?, ?)
                """, (fight_id, odds.open_a_odds, odds.open_b_odds,
                      odds.close_a_odds, odds.close_b_odds))
        
        conn.commit()
        conn.close()


def main():
    """Test historical odds collection."""
    collector = BestFightOddsHistorical()
    
    # Test with a recent event
    test_url = "https://www.bestfightodds.com/events/ufc-fight-night-255-5345"
    odds = collector.fetch_event_odds(test_url)
    
    print(f"Found {len(odds)} fights with odds")
    for o in odds[:3]:
        print(f"{o.fighter_a} ({o.close_a_odds}) vs {o.fighter_b} ({o.close_b_odds})")


if __name__ == "__main__":
    main()
