"""Calendar and business-day calculation engine for ZentroMart.

Rules (per KB-00.2 and KB-00.3):
- Reference date is 20 October 2026.
- Business days: Monday through Saturday (Sunday is excluded).
- Warehouse holidays in data/holidays.csv are excluded from business days.
- Return windows are counted in calendar days starting from the delivery date.
"""
import csv
from datetime import date, timedelta
from typing import Set, Tuple, Union
from src.config import DATA_DIR, REFERENCE_DATE


def load_holidays() -> Set[date]:
    """Load warehouse holidays from data/holidays.csv."""
    holidays = set()
    holidays_file = DATA_DIR / "holidays.csv"
    if holidays_file.exists():
        with open(holidays_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d_str = row.get("date", "").strip()
                if d_str:
                    try:
                        holidays.add(date.fromisoformat(d_str))
                    except ValueError:
                        pass
    else:
        # Fallback to known holidays in KB-00.3
        holidays = {
            date(2026, 8, 15),
            date(2026, 10, 2),
            date(2026, 10, 19),
            date(2026, 11, 9),
        }
    return holidays


HOLIDAYS = load_holidays()


def is_business_day(d: date, holidays: Set[date] = None) -> bool:
    """Return True if the date is Mon-Sat and not a warehouse holiday."""
    if holidays is None:
        holidays = HOLIDAYS
    # weekday(): Mon=0, Tue=1, ..., Sat=5, Sun=6
    if d.weekday() == 6:  # Sunday
        return False
    if d in holidays:
        return False
    return True


def add_business_days(start_date: Union[date, str], days: int, holidays: Set[date] = None) -> date:
    """Add N business days to start_date, counting Mon-Sat and skipping holidays."""
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if holidays is None:
        holidays = HOLIDAYS

    current = start_date
    added = 0
    while added < days:
        current += timedelta(days=1)
        if is_business_day(current, holidays):
            added += 1
    return current


def business_days_between(start_date: Union[date, str], end_date: Union[date, str], holidays: Set[date] = None) -> int:
    """Count business days between start_date (exclusive) and end_date (inclusive)."""
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)
    if holidays is None:
        holidays = HOLIDAYS

    if start_date > end_date:
        return -business_days_between(end_date, start_date, holidays)

    count = 0
    cur = start_date + timedelta(days=1)
    while cur <= end_date:
        if is_business_day(cur, holidays):
            count += 1
        cur += timedelta(days=1)
    return count


def add_calendar_days(start_date: Union[date, str], days: int) -> date:
    """Add calendar days (used for return windows starting from delivery date)."""
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    return start_date + timedelta(days=days)


def format_date_natural(d: date) -> str:
    """Format date naturally (e.g., '27 October 2026')."""
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    return f"{d.day} {months[d.month - 1]} {d.year}"


def format_date_speech(d: date) -> str:
    """Format date for speech ear (e.g., 'the 27th of October 2026')."""
    day = d.day
    if 11 <= (day % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    return f"the {day}{suffix} of {months[d.month - 1]} {d.year}"
