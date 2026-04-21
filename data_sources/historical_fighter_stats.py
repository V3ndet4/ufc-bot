"""
Historical Fighter Stats Builder

Builds fighter stats as of a specific date (fight date).
This is critical for model training — we can't use stats from future fights!

Approach:
- Start from ufcstats.com fighter pages
- Track fight history for each fighter
- Build cumulative stats BEFORE each fight
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "http://www.ufcstats.com"


@dataclass
class FighterSnapshot:
    """Fighter stats at a specific point in time."""
    fighter_name: str
    fight_id: str  # The fight this snapshot is for
    event_date: str
    
    # Physical (these don't change much)
    height_in: Optional[float]
    reach_in: Optional[float]
    stance: str
    
    # Career stats (cumulative BEFORE this fight)
    ufc_wins: int
    ufc_losses: int
    ufc_draws: int
    ufc_ncs: int
    total_ufc_fights: int
    
    # Performance metrics (from UFC Stats)
    sig_strikes_landed_per_min: float
    sig_strikes_absorbed_per_min: float
    takedown_avg: float
    takedown_defense_pct: float
    submission_avg: float
    
    # Derived
    days_since_last_fight: Optional[int]
    finish_rate: float
    decision_rate: float


class HistoricalFighterStatsBuilder:
    """
    Builds historical fighter stats by scraping fight-by-fight history.
    
    This is the HARD part of UFC modeling — you need stats as of fight time,
    not current stats which include the fights you're trying to predict!
    """
    
    def __init__(self, db_path: str = "data/historical_ufc.db", delay: float = 1.0):
        self.db_path = Path(db_path)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def fetch_fighter_page(self, fighter_url: str) -> BeautifulSoup:
        """Fetch a fighter's ufcstats.com page."""
        response = self.session.get(fighter_url)
        response.raise_for_status()
        time.sleep(self.delay)
        return BeautifulSoup(response.text, "html.parser")
    
    def parse_fighter_info(self, soup: BeautifulSoup) -> dict:
        """Extract physical info from fighter page."""
        info = {}
        
        # Physical stats
        height_elem = soup.find("strong", string="Height:")
        if height_elem:
            height_text = height_elem.parent.text.replace("Height:", "").strip()
            info["height"] = self._parse_height(height_text)
        
        reach_elem = soup.find("strong", string="Reach:")
        if reach_elem:
            reach_text = reach_elem.parent.text.replace("Reach:", "").strip()
            info["reach"] = self._parse_reach(reach_text)
        
        stance_elem = soup.find("strong", string="Stance:")
        if stance_elem:
            info["stance"] = stance_elem.parent.text.replace("Stance:", "").strip()
        
        return info
    
    def parse_fight_history(self, soup: BeautifulSoup) -> list[dict]:
        """
        Extract fight history from fighter page.
        Returns fights in chronological order (oldest first).
        """
        fights = []
        
        # Fight history table
        rows = soup.select("tbody tr")
        
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 6:
                continue
            
            # Result
            result = cells[0].text.strip()
            
            # Opponent
            opponent = cells[1].text.strip()
            opponent_link = cells[1].find("a", href=True)
            opponent_url = opponent_link["href"] if opponent_link else None
            
            # Event
            event = cells[2].text.strip()
            event_link = cells[2].find("a", href=True)
            event_url = event_link["href"] if event_link else None
            
            # Method, round, time
            method = cells[3].text.strip()
            round_num = cells[4].text.strip()
            time_str = cells[5].text.strip()
            
            fights.append({
                "result": result,
                "opponent": opponent,
                "opponent_url": opponent_url,
                "event": event,
                "event_url": event_url,
                "method": method,
                "round": round_num,
                "time": time_str,
                "is_win": result.upper() == "W",
                "is_loss": result.upper() == "L",
                "is_draw": result.upper() == "D",
                "is_nc": result.upper() == "NC"
            })
        
        return fights
    
    def build_cumulative_stats(
        self, 
        fighter_name: str,
        fight_history: list[dict],
        target_fight_index: int
    ) -> dict:
        """
        Build stats as of a specific fight.
        
        fight_history: chronological list of all UFC fights
        target_fight_index: which fight we're building stats for
                          (0 = first fight, stats will be mostly empty)
        """
        # Only use fights BEFORE target fight
        prior_fights = fight_history[:target_fight_index]
        
        stats = {
            "ufc_wins": sum(1 for f in prior_fights if f["is_win"]),
            "ufc_losses": sum(1 for f in prior_fights if f["is_loss"]),
            "ufc_draws": sum(1 for f in prior_fights if f["is_draw"]),
            "ufc_ncs": sum(1 for f in prior_fights if f["is_nc"]),
            "total_ufc_fights": len(prior_fights),
            "finish_rate": 0.0,
            "decision_rate": 0.0,
            "sig_strikes_landed_per_min": 0.0,
            "sig_strikes_absorbed_per_min": 0.0,
            "takedown_avg": 0.0,
            "takedown_defense_pct": 0.0
        }
        
        # Calculate finish/decision rates from prior fights
        if prior_fights:
            finishes = sum(1 for f in prior_fights 
                          if "KO" in f["method"] or "TKO" in f["method"] 
                          or "Sub" in f["method"])
            decisions = sum(1 for f in prior_fights if "Decision" in f["method"])
            
            stats["finish_rate"] = finishes / len(prior_fights)
            stats["decision_rate"] = decisions / len(prior_fights)
        
        # Days since last fight
        if target_fight_index > 0 and target_fight_index < len(fight_history):
            # Would need to parse actual dates from event pages
            # For now, placeholder
            stats["days_since_last_fight"] = None
        
        return stats
    
    def _parse_height(self, height_str: str) -> Optional[float]:
        """Parse height like '6\' 2"' to inches."""
        try:
            parts = height_str.replace('"', '').split("'")
            feet = int(parts[0].strip())
            inches = int(parts[1].strip()) if len(parts) > 1 else 0
            return feet * 12 + inches
        except:
            return None
    
    def _parse_reach(self, reach_str: str) -> Optional[float]:
        """Parse reach like '72.0"' to inches."""
        try:
            return float(reach_str.replace('"', '').strip())
        except:
            return None
    
    def get_fighter_snapshot(
        self,
        fighter_name: str,
        fighter_url: str,
        event_date: str,
        as_of_date: str
    ) -> Optional[FighterSnapshot]:
        """
        Get fighter stats as of a specific date.
        
        This requires:
        1. Scraping the fighter's full history
        2. Finding which fight occurred just before as_of_date
        3. Building cumulative stats from all prior fights
        """
        try:
            soup = self.fetch_fighter_page(fighter_url)
            info = self.parse_fighter_info(soup)
            history = self.parse_fight_history(soup)
            
            # Find fight index (simplified - would need proper date parsing)
            fight_index = 0  # Placeholder
            
            cumulative = self.build_cumulative_stats(fighter_name, history, fight_index)
            
            return FighterSnapshot(
                fighter_name=fighter_name,
                fight_id="",  # Would match from fights table
                event_date=event_date,
                height_in=info.get("height"),
                reach_in=info.get("reach"),
                stance=info.get("stance", ""),
                ufc_wins=cumulative["ufc_wins"],
                ufc_losses=cumulative["ufc_losses"],
                ufc_draws=cumulative["ufc_draws"],
                ufc_ncs=cumulative["ufc_ncs"],
                total_ufc_fights=cumulative["total_ufc_fights"],
                sig_strikes_landed_per_min=cumulative["sig_strikes_landed_per_min"],
                sig_strikes_absorbed_per_min=cumulative["sig_strikes_absorbed_per_min"],
                takedown_avg=cumulative["takedown_avg"],
                takedown_defense_pct=cumulative["takedown_defense_pct"],
                submission_avg=0.0,
                days_since_last_fight=cumulative.get("days_since_last_fight"),
                finish_rate=cumulative["finish_rate"],
                decision_rate=cumulative["decision_rate"]
            )
        
        except Exception as e:
            print(f"Error getting snapshot for {fighter_name}: {e}")
            return None


def main():
    """Test historical fighter stats."""
    builder = HistoricalFighterStatsBuilder()
    
    # Test with a known fighter URL
    test_url = "http://www.ufcstats.com/fighter-details/fighter-name-here"
    
    # This would be called during full data collection
    print("Historical fighter stats builder ready")


if __name__ == "__main__":
    main()
