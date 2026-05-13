"""
UFC Stats Scraper - Real Implementation

Scrapes fighter and fight data from ufcstats.com
This is the real data source for model training.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.ufcstats.com"


@dataclass
class FighterProfile:
    """Complete fighter profile from UFC Stats."""
    name: str
    url: str
    height: Optional[float]  # inches
    weight: Optional[float]  # lbs
    reach: Optional[float]  # inches
    stance: str
    dob: Optional[str]  # YYYY-MM-DD
    wins: int
    losses: int
    draws: int


@dataclass
class FightDetail:
    """Detailed fight information."""
    event_name: str
    event_date: str
    event_location: str
    opponent: str
    result: str  # W, L, D, NC
    method: str
    round: int
    time: str
    
    # Striking stats (per fight)
    sig_strikes_landed: int
    sig_strikes_attempted: int
    head_strikes: int
    body_strikes: int
    leg_strikes: int
    
    # Grappling stats
    takedowns_landed: int
    takedowns_attempted: int
    submissions_attempted: int


class UFCStatsScraper:
    """Scraper for ufcstats.com - the official UFC stats source."""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
    
    def get_event_list(self) -> list[dict]:
        """Get all UFC events."""
        url = f"{BASE_URL}/statistics/events/completed?page=all"
        print(f"Fetching event list from {url}")
        
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        events = []
        
        # Events are in table rows
        for row in soup.select("tr.b-statistics__table-row"):
            link = row.select_one("a.b-link")
            if not link:
                continue
            
            event_url = link.get("href", "")
            event_name = link.text.strip()
            
            # Get date from span
            date_span = row.select_one("span.b-statistics__date")
            event_date = date_span.text.strip() if date_span else ""
            
            events.append({
                "name": event_name,
                "url": event_url,
                "date": event_date,
                "id": event_url.split("/")[-1] if event_url else ""
            })
        
        print(f"Found {len(events)} events")
        return events
    
    def get_event_fights(self, event_url: str) -> list[dict]:
        """Get all fights from an event."""
        print(f"Scraping event: {event_url}")
        
        response = self.session.get(event_url, timeout=30)
        response.raise_for_status()
        time.sleep(self.delay)
        
        soup = BeautifulSoup(response.text, "html.parser")
        fights = []
        
        # Get event info
        event_name = soup.select_one("h2.b-content__title")
        event_name = event_name.text.strip() if event_name else "Unknown"
        
        # Fight table rows
        fight_rows = soup.select("tbody tr.b-fight-details__table-row")
        
        for row in fight_rows:
            fight = self._parse_fight_row(row, event_name)
            if fight:
                fights.append(fight)
        
        return fights
    
    def _parse_fight_row(self, row, event_name: str) -> Optional[dict]:
        """Parse a single fight row."""
        tds = row.find_all("td")
        if len(tds) < 5:
            return None
        
        # Result column - indicates if first fighter won
        result_texts = [
            value.strip()
            for value in [cell.get_text(" ", strip=True) for cell in tds[0].select("p.b-fight-details__table-text")]
            if value.strip()
        ]
        
        # Fighter names
        fighter_links = tds[1].select("p.b-fight-details__table-text a")
        if len(fighter_links) < 2:
            return None
        
        fighter_a = fighter_links[0].text.strip()
        fighter_b = fighter_links[1].text.strip()
        
        # Determine winner based on result column
        winner = None
        lowered_results = [text.lower() for text in result_texts]
        if any(text in {"draw", "d"} or "draw" in text for text in lowered_results):
            winner = "DRAW"
        elif any(text in {"nc", "n/c"} or "no contest" in text for text in lowered_results):
            winner = "NC"
        elif len(lowered_results) >= 2:
            first_result = lowered_results[0]
            second_result = lowered_results[1]
            if _is_win_marker(first_result) or _is_loss_marker(second_result):
                winner = fighter_a
            elif _is_win_marker(second_result) or _is_loss_marker(first_result):
                winner = fighter_b
        elif lowered_results:
            first_result = lowered_results[0]
            if _is_win_marker(first_result):
                winner = fighter_a
        
        # Method
        method = _table_cell_text(tds[-3]) or "Unknown"

        # Round
        round_num = _table_cell_text(tds[-2]) or "0"

        # Time
        time_str = _table_cell_text(tds[-1])
        
        return {
            "event_name": event_name,
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "winner": winner,
            "method": method,
            "round": round_num,
            "time": time_str,
        }


def _is_win_marker(value: str) -> bool:
    cleaned = value.strip().lower()
    return cleaned in {"w", "win"} or "win" in cleaned


def _is_loss_marker(value: str) -> bool:
    cleaned = value.strip().lower()
    return cleaned in {"l", "loss"} or "loss" in cleaned


def _table_cell_text(cell) -> str:
    texts = [" ".join(text.split()) for text in cell.stripped_strings if str(text).strip()]
    cleaned = " ".join(texts)
    cleaned = re.sub(r"\bImage\b", "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split()).strip()
    
    def get_fighter_profile(self, fighter_url: str) -> Optional[FighterProfile]:
        """Get complete fighter profile."""
        print(f"Scraping fighter: {fighter_url}")
        
        response = self.session.get(fighter_url)
        response.raise_for_status()
        time.sleep(self.delay)
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Name
        name_elem = soup.select_one("span.b-content__title-highlight")
        name = name_elem.text.strip() if name_elem else "Unknown"
        
        # Physical stats
        height = self._parse_stat(soup, "Height:")
        weight = self._parse_stat(soup, "Weight:")
        reach = self._parse_stat(soup, "Reach:")
        stance = self._parse_text_stat(soup, "Stance:")
        
        # Record
        record_elem = soup.select_one("span.b-content__title-record")
        wins, losses, draws = 0, 0, 0
        if record_elem:
            record_text = record_elem.text.strip()
            match = re.search(r'(\d+)-(\d+)-(\d+)', record_text)
            if match:
                wins, losses, draws = int(match[1]), int(match[2]), int(match[3])
        
        return FighterProfile(
            name=name,
            url=fighter_url,
            height=self._height_to_inches(height),
            weight=self._weight_to_lbs(weight),
            reach=self._reach_to_inches(reach),
            stance=stance or "",
            dob=None,  # Would need additional parsing
            wins=wins,
            losses=losses,
            draws=draws
        )
    
    def get_fighter_fight_history(self, fighter_url: str) -> list[FightDetail]:
        """Get detailed fight history for a fighter."""
        print(f"Scraping fight history: {fighter_url}")
        
        response = self.session.get(fighter_url)
        response.raise_for_status()
        time.sleep(self.delay)
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        fights = []
        history_rows = soup.select("tbody tr")
        
        for row in history_rows:
            fight = self._parse_history_row(row)
            if fight:
                fights.append(fight)
        
        return fights
    
    def _parse_stat(self, soup: BeautifulSoup, label: str) -> str:
        """Parse a stat value by label."""
        elem = soup.find("i", string=lambda t: t and label in t)
        if elem:
            parent = elem.parent
            if parent:
                text = parent.get_text().replace(label, "").strip()
                return text
        return ""
    
    def _parse_text_stat(self, soup: BeautifulSoup, label: str) -> str:
        """Parse a text stat value."""
        elem = soup.find("i", string=lambda t: t and label in t)
        if elem:
            next_sib = elem.find_next_sibling(string=True)
            if next_sib:
                return next_sib.strip()
        return ""
    
    def _height_to_inches(self, height_str: str) -> Optional[float]:
        """Convert height like '6' 2"' to inches."""
        if not height_str:
            return None
        try:
            # Format: 6' 2" or 6'2"
            match = re.search(r"(\d+)'\s*(\d+)?", height_str)
            if match:
                feet = int(match[1])
                inches = int(match[2]) if match[2] else 0
                return feet * 12 + inches
        except:
            pass
        return None
    
    def _weight_to_lbs(self, weight_str: str) -> Optional[float]:
        """Extract weight in lbs."""
        if not weight_str:
            return None
        try:
            match = re.search(r'(\d+)', weight_str)
            if match:
                return float(match[1])
        except:
            pass
        return None
    
    def _reach_to_inches(self, reach_str: str) -> Optional[float]:
        """Convert reach like '72.0"' to inches."""
        if not reach_str:
            return None
        try:
            match = re.search(r'([\d.]+)', reach_str)
            if match:
                return float(match[1])
        except:
            pass
        return None


def test_scraper():
    """Test the scraper."""
    scraper = UFCStatsScraper(delay=1.0)
    
    # Test getting events
    events = scraper.get_event_list()
    print(f"\nFirst 3 events:")
    for e in events[:3]:
        print(f"  {e['date']}: {e['name']}")
    
    # Test getting fights from most recent event
    if events:
        print(f"\nScraping fights from: {events[0]['name']}")
        fights = scraper.get_event_fights(events[0]['url'])
        print(f"Found {len(fights)} fights")
        for f in fights[:2]:
            print(f"  {f['fighter_a']} vs {f['fighter_b']}: {f['winner']} wins by {f['method']}")


if __name__ == "__main__":
    test_scraper()
