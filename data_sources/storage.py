from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from backtests.grading import attach_tracked_expression_columns, grade_tracked_picks

ODDS_SNAPSHOT_COLUMNS = {
    "event_id": "TEXT NOT NULL",
    "event_name": "TEXT NOT NULL",
    "start_time": "TEXT NOT NULL",
    "fighter_a": "TEXT NOT NULL",
    "fighter_b": "TEXT NOT NULL",
    "market": "TEXT NOT NULL",
    "selection": "TEXT NOT NULL",
    "selection_name": "TEXT",
    "book": "TEXT NOT NULL",
    "american_odds": "INTEGER NOT NULL",
    "projected_win_prob": "REAL",
}

TRACKED_PICK_COLUMNS = {
    "event_id": "TEXT NOT NULL",
    "event_name": "TEXT NOT NULL",
    "start_time": "TEXT NOT NULL",
    "fighter_a": "TEXT NOT NULL",
    "fighter_b": "TEXT NOT NULL",
    "market": "TEXT NOT NULL",
    "selection": "TEXT NOT NULL",
    "selection_name": "TEXT",
    "book": "TEXT NOT NULL",
    "american_odds": "INTEGER NOT NULL",
    "model_projected_win_prob": "REAL",
    "implied_prob": "REAL",
    "edge": "REAL",
    "expected_value": "REAL",
    "suggested_stake": "REAL",
    "raw_suggested_stake": "REAL",
    "model_confidence": "REAL",
    "data_quality": "REAL",
    "selection_stats_completeness": "REAL",
    "selection_fallback_used": "REAL",
    "line_movement_toward_fighter": "REAL",
    "market_blend_weight": "REAL",
    "market_consensus_bookmaker_count": "REAL",
    "market_overround": "REAL",
    "price_edge_vs_consensus": "REAL",
    "bet_quality_score": "REAL",
    "support_count": "REAL",
    "risk_flag_count": "REAL",
    "recommended_tier": "TEXT",
    "recommended_action": "TEXT",
    "support_signals": "TEXT",
    "risk_flags": "TEXT",
    "timing_snapshot_count": "REAL",
    "timing_book_count": "REAL",
    "timing_open_implied_prob": "REAL",
    "timing_latest_implied_prob": "REAL",
    "timing_implied_change": "REAL",
    "timing_velocity_per_hour": "REAL",
    "timing_volatility": "REAL",
    "timing_book_dispersion": "REAL",
    "timing_score": "REAL",
    "timing_signal": "TEXT",
    "timing_action": "TEXT",
    "timing_reason": "TEXT",
    "news_alert_count": "REAL",
    "news_radar_score": "REAL",
    "news_radar_label": "TEXT",
    "news_radar_summary": "TEXT",
    "scheduled_rounds": "REAL",
    "is_wmma": "REAL",
    "is_heavyweight": "REAL",
    "is_five_round_fight": "REAL",
    "segment_label": "TEXT",
    "selection_recent_finish_damage": "REAL",
    "selection_recent_ko_damage": "REAL",
    "selection_recent_damage_score": "REAL",
    "selection_recent_grappling_rate": "REAL",
    "selection_control_avg": "REAL",
    "selection_recent_control_avg": "REAL",
    "selection_gym_score": "REAL",
    "selection_context_instability": "REAL",
    "selection_matchup_striking_edge": "REAL",
    "selection_matchup_grappling_edge": "REAL",
    "selection_matchup_control_edge": "REAL",
    "selection_stance_matchup_edge": "REAL",
    "base_model_projected_win_prob": "REAL",
    "trained_side_selection_prob": "REAL",
    "trained_side_fighter_a_win_prob": "REAL",
    "side_model_blend_weight": "REAL",
    "selective_clv_prob": "REAL",
    "chosen_value_expression": "TEXT",
    "expression_pick_source": "TEXT",
    "chosen_expression_odds": "INTEGER",
    "chosen_expression_prob": "REAL",
    "chosen_expression_implied_prob": "REAL",
    "chosen_expression_edge": "REAL",
    "chosen_expression_expected_value": "REAL",
    "chosen_expression_stake": "REAL",
    "raw_chosen_expression_stake": "REAL",
    "stake_governor_multiplier": "REAL",
    "stake_cap_per_bet": "REAL",
    "stake_cap_per_fight": "REAL",
    "stake_cap_per_card": "REAL",
    "stake_governor_reason": "TEXT",
    "tracked_market_key": "TEXT",
    "tracked_selection_key": "TEXT",
    "fight_key": "TEXT",
    "closing_american_odds": "INTEGER",
    "clv_delta": "REAL",
    "clv_edge": "REAL",
    "actual_result": "TEXT",
    "profit": "REAL",
    "grade_status": "TEXT DEFAULT 'pending'",
}

FIGHT_RESULT_COLUMNS = {
    "event_id": "TEXT NOT NULL",
    "event_name": "TEXT",
    "fighter_a": "TEXT NOT NULL",
    "fighter_b": "TEXT NOT NULL",
    "winner_name": "TEXT",
    "winner_side": "TEXT",
    "result_status": "TEXT",
    "went_decision": "INTEGER",
    "ended_inside_distance": "INTEGER",
    "method": "TEXT",
    "closing_fighter_a_odds": "INTEGER",
    "closing_fighter_b_odds": "INTEGER",
    "closing_fight_goes_to_decision_odds": "INTEGER",
    "closing_fight_doesnt_go_to_decision_odds": "INTEGER",
    "closing_fighter_a_inside_distance_odds": "INTEGER",
    "closing_fighter_b_inside_distance_odds": "INTEGER",
    "closing_fighter_a_by_decision_odds": "INTEGER",
    "closing_fighter_b_by_decision_odds": "INTEGER",
    "closing_fighter_a_submission_odds": "INTEGER",
    "closing_fighter_b_submission_odds": "INTEGER",
    "closing_fighter_a_ko_tko_odds": "INTEGER",
    "closing_fighter_b_ko_tko_odds": "INTEGER",
    "closing_fight_ends_by_submission_odds": "INTEGER",
    "closing_fight_ends_by_ko_tko_odds": "INTEGER",
    "closing_fighter_a_knockdown_odds": "INTEGER",
    "closing_fighter_b_knockdown_odds": "INTEGER",
    "closing_fighter_a_takedown_odds": "INTEGER",
    "closing_fighter_b_takedown_odds": "INTEGER",
    "fighter_a_knockdowns": "REAL",
    "fighter_b_knockdowns": "REAL",
    "fighter_a_takedowns": "REAL",
    "fighter_b_takedowns": "REAL",
}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_snapshot_columns(connection: sqlite3.Connection) -> None:
    existing = {
        row[1]
        for row in connection.execute("PRAGMA table_info(odds_snapshots)").fetchall()
    }
    for column_name, column_type in ODDS_SNAPSHOT_COLUMNS.items():
        if column_name not in existing:
            connection.execute(f"ALTER TABLE odds_snapshots ADD COLUMN {column_name} {column_type}")
    connection.commit()


def _ensure_table_columns(connection: sqlite3.Connection, table_name: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}
    for column_name, column_type in columns.items():
        if column_name not in existing:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
    connection.commit()


def init_db(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    _ensure_parent(path)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            event_name TEXT NOT NULL,
            start_time TEXT NOT NULL,
            fighter_a TEXT NOT NULL,
            fighter_b TEXT NOT NULL,
            market TEXT NOT NULL,
            selection TEXT NOT NULL,
            selection_name TEXT,
            book TEXT NOT NULL,
            american_odds INTEGER NOT NULL,
            projected_win_prob REAL,
            snapshot_time TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _ensure_snapshot_columns(connection)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            picks INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            total_staked REAL NOT NULL,
            total_profit REAL NOT NULL,
            roi REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tracked_picks (
            pick_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            event_name TEXT NOT NULL,
            start_time TEXT NOT NULL,
            fighter_a TEXT NOT NULL,
            fighter_b TEXT NOT NULL,
            market TEXT NOT NULL,
            selection TEXT NOT NULL,
            selection_name TEXT,
            book TEXT NOT NULL,
            american_odds INTEGER NOT NULL,
            model_projected_win_prob REAL,
            implied_prob REAL,
            edge REAL,
            expected_value REAL,
            suggested_stake REAL,
            model_confidence REAL,
            data_quality REAL,
            selection_stats_completeness REAL,
            selection_fallback_used REAL,
            line_movement_toward_fighter REAL,
            market_blend_weight REAL,
            bet_quality_score REAL,
            recommended_tier TEXT,
            recommended_action TEXT,
            support_signals TEXT,
            risk_flags TEXT,
            timing_snapshot_count REAL,
            timing_book_count REAL,
            timing_open_implied_prob REAL,
            timing_latest_implied_prob REAL,
            timing_implied_change REAL,
            timing_velocity_per_hour REAL,
            timing_volatility REAL,
            timing_book_dispersion REAL,
            timing_score REAL,
            timing_signal TEXT,
            timing_action TEXT,
            timing_reason TEXT,
            news_alert_count REAL,
            news_radar_score REAL,
            news_radar_label TEXT,
            news_radar_summary TEXT,
            chosen_value_expression TEXT,
            expression_pick_source TEXT,
            chosen_expression_odds INTEGER,
            chosen_expression_prob REAL,
            chosen_expression_implied_prob REAL,
            chosen_expression_edge REAL,
            chosen_expression_expected_value REAL,
            chosen_expression_stake REAL,
            tracked_market_key TEXT,
            tracked_selection_key TEXT,
            fight_key TEXT,
            closing_american_odds INTEGER,
            clv_delta REAL,
            clv_edge REAL,
            actual_result TEXT,
            profit REAL,
            grade_status TEXT DEFAULT 'pending',
            tracked_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS fight_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            event_name TEXT,
            fighter_a TEXT NOT NULL,
            fighter_b TEXT NOT NULL,
            winner_name TEXT,
            winner_side TEXT,
            result_status TEXT,
            went_decision INTEGER,
            ended_inside_distance INTEGER,
            method TEXT,
            closing_fighter_a_odds INTEGER,
            closing_fighter_b_odds INTEGER,
            closing_fight_goes_to_decision_odds INTEGER,
            closing_fight_doesnt_go_to_decision_odds INTEGER,
            closing_fighter_a_inside_distance_odds INTEGER,
            closing_fighter_b_inside_distance_odds INTEGER,
            closing_fighter_a_by_decision_odds INTEGER,
            closing_fighter_b_by_decision_odds INTEGER,
            closing_fighter_a_submission_odds INTEGER,
            closing_fighter_b_submission_odds INTEGER,
            closing_fighter_a_ko_tko_odds INTEGER,
            closing_fighter_b_ko_tko_odds INTEGER,
            closing_fight_ends_by_submission_odds INTEGER,
            closing_fight_ends_by_ko_tko_odds INTEGER,
            closing_fighter_a_knockdown_odds INTEGER,
            closing_fighter_b_knockdown_odds INTEGER,
            closing_fighter_a_takedown_odds INTEGER,
            closing_fighter_b_takedown_odds INTEGER,
            fighter_a_knockdowns REAL,
            fighter_b_knockdowns REAL,
            fighter_a_takedowns REAL,
            fighter_b_takedowns REAL,
            imported_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.commit()
    _ensure_table_columns(connection, "tracked_picks", TRACKED_PICK_COLUMNS)
    _ensure_table_columns(connection, "fight_results", FIGHT_RESULT_COLUMNS)
    return connection


def save_odds_snapshot(frame: pd.DataFrame, db_path: str | Path) -> int:
    connection = init_db(db_path)
    try:
        ordered_columns = [column for column in ODDS_SNAPSHOT_COLUMNS if column in frame.columns]
        frame.loc[:, ordered_columns].to_sql("odds_snapshots", connection, if_exists="append", index=False)
        connection.commit()
        return int(len(frame))
    finally:
        connection.close()


def load_snapshot_history(db_path: str | Path, event_id: str | None = None) -> pd.DataFrame:
    connection = init_db(db_path)
    query = "SELECT * FROM odds_snapshots"
    params: tuple[str, ...] = ()
    if event_id:
        query += " WHERE event_id = ?"
        params = (event_id,)
    try:
        return pd.read_sql_query(query, connection, params=params)
    finally:
        connection.close()


def save_backtest_run(summary: dict[str, float | int], db_path: str | Path) -> None:
    connection = init_db(db_path)
    try:
        connection.execute(
            """
            INSERT INTO backtest_runs (picks, wins, losses, total_staked, total_profit, roi)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                summary["picks"],
                summary["wins"],
                summary["losses"],
                summary["total_staked"],
                summary["total_profit"],
                summary["roi"],
            ),
        )
        connection.commit()
    finally:
        connection.close()


def save_tracked_picks(frame: pd.DataFrame, db_path: str | Path) -> int:
    connection = init_db(db_path)
    try:
        actionable = frame.copy()
        if "recommended_action" in actionable.columns:
            actionable = actionable.loc[actionable["recommended_action"].isin(["Bettable now", "Watchlist"])].copy()
        if actionable.empty:
            return 0
        enriched = attach_tracked_expression_columns(actionable)
        key_columns = ["event_id", "fight_key", "tracked_market_key", "tracked_selection_key"]
        enriched = enriched.drop_duplicates(subset=key_columns, keep="last")
        rows_to_insert: list[dict[str, object]] = []
        for row in enriched.to_dict("records"):
            key_values = tuple(str(row.get(column, "") or "") for column in key_columns)
            existing_graded = connection.execute(
                """
                SELECT 1
                FROM tracked_picks
                WHERE event_id = ?
                  AND fight_key = ?
                  AND tracked_market_key = ?
                  AND tracked_selection_key = ?
                  AND LOWER(COALESCE(grade_status, 'pending')) = 'graded'
                LIMIT 1
                """,
                key_values,
            ).fetchone()
            if existing_graded:
                continue
            connection.execute(
                """
                DELETE FROM tracked_picks
                WHERE event_id = ?
                  AND fight_key = ?
                  AND tracked_market_key = ?
                  AND tracked_selection_key = ?
                  AND LOWER(COALESCE(grade_status, 'pending')) != 'graded'
                """,
                key_values,
            )
            rows_to_insert.append(row)
        if not rows_to_insert:
            connection.commit()
            return 0
        enriched = pd.DataFrame.from_records(rows_to_insert)
        ordered_columns = [column for column in TRACKED_PICK_COLUMNS if column in enriched.columns]
        enriched.loc[:, ordered_columns].to_sql("tracked_picks", connection, if_exists="append", index=False)
        connection.commit()
        return int(len(enriched))
    finally:
        connection.close()


def load_tracked_picks(db_path: str | Path, event_id: str | None = None) -> pd.DataFrame:
    connection = init_db(db_path)
    query = "SELECT * FROM tracked_picks"
    params: tuple[str, ...] = ()
    if event_id:
        query += " WHERE event_id = ?"
        params = (event_id,)
    try:
        return pd.read_sql_query(query, connection, params=params)
    finally:
        connection.close()


def save_fight_results(frame: pd.DataFrame, db_path: str | Path) -> int:
    connection = init_db(db_path)
    try:
        if frame.empty:
            return 0
        ordered_columns = [column for column in FIGHT_RESULT_COLUMNS if column in frame.columns]
        event_ids = [str(value) for value in frame["event_id"].dropna().unique()]
        for event_id in event_ids:
            connection.execute("DELETE FROM fight_results WHERE event_id = ?", (event_id,))
        frame.loc[:, ordered_columns].to_sql("fight_results", connection, if_exists="append", index=False)
        connection.commit()
        return int(len(frame))
    finally:
        connection.close()


def load_fight_results(db_path: str | Path, event_id: str | None = None) -> pd.DataFrame:
    connection = init_db(db_path)
    query = "SELECT * FROM fight_results"
    params: tuple[str, ...] = ()
    if event_id:
        query += " WHERE event_id = ?"
        params = (event_id,)
    try:
        return pd.read_sql_query(query, connection, params=params)
    finally:
        connection.close()


def update_tracked_pick_grades(graded_frame: pd.DataFrame, db_path: str | Path) -> None:
    connection = init_db(db_path)
    try:
        for row in graded_frame.to_dict("records"):
            if "pick_id" not in row:
                continue
            connection.execute(
                """
                UPDATE tracked_picks
                SET closing_american_odds = ?,
                    clv_delta = ?,
                    clv_edge = ?,
                    actual_result = ?,
                    profit = ?,
                    grade_status = ?
                WHERE pick_id = ?
                """,
                (
                    row.get("closing_american_odds"),
                    row.get("clv_delta"),
                    row.get("clv_edge"),
                    row.get("actual_result"),
                    row.get("profit"),
                    row.get("grade_status", "graded"),
                    row["pick_id"],
                ),
            )
        connection.commit()
    finally:
        connection.close()


def grade_pending_picks(db_path: str | Path, event_id: str | None = None) -> pd.DataFrame:
    picks = load_tracked_picks(db_path, event_id=event_id)
    if picks.empty:
        return picks
    pending = picks.loc[picks["grade_status"].fillna("pending") == "pending"].copy()
    if pending.empty:
        return pending
    results = load_fight_results(db_path, event_id=event_id)
    if results.empty:
        return pending.iloc[0:0].copy()
    graded = grade_tracked_picks(pending, results)
    completed = graded.loc[graded["actual_result"] != "pending"].copy()
    if not completed.empty:
        update_tracked_pick_grades(completed, db_path)
    return completed
