#!/usr/bin/env python3
"""
Fubon ETF Ultimate Fast PCF Scraper
Maximum speed + timeout bypass + intelligent retry
Usage giống hệt bản UltimateFastScraper gốc của mày
"""

import os
from pathlib import Path
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


class UltimateFastPCFScraper:
    def __init__(self, etf_code="00885", output_dir=None):
        self.etf_code = etf_code
        # Use project-relative default if not provided
        if output_dir is None:
            output_dir = ROOT / "data" / "fubon_weight_data" / "portfolio_composition"
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.file_lock = threading.Lock()
        self.seen_data_dates = {}
        self.seen_lock = threading.Lock()

        import urllib3
        urllib3.disable_warnings()

        self.session_pool = []
        for _ in range(30):
            self.session_pool.append(self._create_fast_session())

        logger.info(f"🚀 Ultimate Fast PCF Scraper initialized for ETF {etf_code}")

    def _create_fast_session(self):
        session = requests.Session()
        session.verify = False
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'text/html,application/xhtml+xml',
        })
        retry = Retry(total=2, backoff_factor=0.1,
                      status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry,
                              pool_connections=50, pool_maxsize=50)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _extract_pcf_data(self, html):
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(" ", strip=True)

        data = {}
        patterns = {
            "total_advance_subscription": r"Amount of Total Advance Subscription[^\d]+([\d,]+)",
            "nav": r"Net Asset Value[^\d]+([\d,]+)",
            "total_units": r"Total Units Outstanding[^\d]+([\d,]+)",
            "net_unit_change": r"Net Unit Change[^\d-]+([-\d,]+)",
            "nav_per_unit": r"NAV Per Unit[^\d]+([\d.]+)",
            "creation_unit": r"Creation/Redemption Unit[^\d]+([\d,]+)",
            "equity_value_basket": r"Equity Value Per Basket[^\d]+([-\d,]+)",
            "cash_component": r"Cash Component Per Basket[^\d-]+([-\d,]+)",
            "price_per_basket": r"Price per Creation Basket[^\d]+([-\d,]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                data[key] = match.group(1).replace(",", "")

        data["fund_purchase"] = "Yes" if "Fund Purchase Yes" in text else "No"
        data["fund_redemption"] = "Yes" if "Fund Redemption Yes" in text else "No"

        date_match = re.search(r"Date:\s*(\d{4}/\d{2}/\d{2})", text)
        if date_match:
            data_date = date_match.group(1)
        else:
            data_date = None

        return data, data_date

    def scrape_date_ultra_fast(self, date_obj, session_idx=0, skip_duplicates=True):
        date_str = date_obj.strftime("%Y%m%d")
        date_display = date_obj.strftime("%Y-%m-%d")

        if skip_duplicates and date_obj.weekday() >= 5:
            return (date_display, "skipped", "weekend")

        url = f"https://websys.fsit.com.tw/FubonETF/Trade/Pcf.aspx?stkId={self.etf_code}&ddate={date_str}&lan=EN"
        session = self.session_pool[session_idx % len(self.session_pool)]

        timeouts = [3, 5, 10]
        for attempt, timeout in enumerate(timeouts):
            try:
                resp = session.get(url, timeout=timeout)
                if resp.status_code == 200:
                    data, data_date = self._extract_pcf_data(resp.text)
                    if not data:
                        continue
                    if not data_date:
                        data_date = date_display

                    with self.seen_lock:
                        if skip_duplicates and data_date in self.seen_data_dates:
                            return (date_display, "duplicate", data_date)
                        self.seen_data_dates[data_date] = date_display

                    df = pd.DataFrame([data])
                    df["requested_date"] = date_display
                    df["data_date"] = data_date
                    df["etf_code"] = self.etf_code

                    csv_file = os.path.join(self.output_dir, f"{self.etf_code}_{data_date.replace('/', '')}.csv")
                    with self.file_lock:
                        if not os.path.exists(csv_file):
                            df.to_csv(csv_file, index=False)

                    return (date_display, "success", {"nav": data.get("nav_per_unit", "N/A"), "timeout": timeout})

                elif resp.status_code == 404:
                    return (date_display, "failed", "404")
                else:
                    continue
            except requests.exceptions.Timeout:
                if attempt < len(timeouts) - 1:
                    continue
                return (date_display, "timeout", "all timeouts failed")
            except Exception as e:
                if attempt < len(timeouts) - 1:
                    continue
                return (date_display, "error", str(e)[:50])

        return (date_display, "failed", "all attempts failed")

    def get_existing_dates(self):
        existing = set()
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                if f.startswith(f"{self.etf_code}_") and f.endswith(".csv"):
                    date_part = f.split("_")[1].split(".")[0]
                    if len(date_part) == 8 and date_part.isdigit():
                        existing.add(f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}")
        return existing

    def scrape_range_ultimate(self, start_date="2021-04-16", end_date=None, max_workers=25, resume=True):
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        existing = set()
        if resume:
            existing = self.get_existing_dates()

        dates_to_process = []
        current = start
        while current <= end:
            if not (resume and current.strftime("%Y-%m-%d") in existing):
                dates_to_process.append(current)
            current += timedelta(days=1)

        total = len(dates_to_process)
        if total == 0:
            logger.info("✅ All dates already downloaded!")
            return {"successful": [], "failed": []}

        successful = []
        failed = []
        duplicates = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.scrape_date_ultra_fast, date, idx, True): (date, idx)
                for idx, date in enumerate(dates_to_process)
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                date, idx = futures[future]
                try:
                    date_str, status, result = future.result()
                    if status == "success":
                        successful.append(date_str)
                        print(f"[{completed}/{total}] {date_str} ✓ NAV: {result['nav']} | {result['timeout']}s")
                    elif status == "duplicate":
                        duplicates.append(date_str)
                        print(f"[{completed}/{total}] {date_str} ⚠ Dup")
                    elif status == "skipped":
                        print(f"[{completed}/{total}] {date_str} - Skip")
                    else:
                        failed.append(date_str)
                        print(f"[{completed}/{total}] {date_str} ✗ {status}")
                except Exception as e:
                    failed.append(date.strftime("%Y-%m-%d"))
                    print(f"[{completed}/{total}] Error: {str(e)[:30]}")

        # Retry failed
        if failed and len(failed) <= 20:
            logger.info(f"\n⚡ Quick retry for {len(failed)} failed dates...")
            for date_str in failed[:]:
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    result = self.scrape_date_ultra_fast(date_obj, 0, True)
                    if result[1] == "success":
                        successful.append(date_str)
                        failed.remove(date_str)
                        print(f"  {date_str} ✓ Recovered")
                except:
                    pass

        # Save failed
        if failed:
            failed_file = os.path.join(self.output_dir, f"{self.etf_code}_failed.txt")
            with open(failed_file, "w") as f:
                for d in failed:
                    f.write(f"{d}\n")
            logger.info(f"📝 Failed dates saved: {failed_file}")

        logger.info(f"✅ Successful: {len(successful)} | ❌ Failed: {len(failed)} | ⚠ Dup: {len(duplicates)}")

        return {"successful": successful, "failed": failed}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="🚀 Ultimate Fast PCF Scraper")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, help="Last N days")
    parser.add_argument("--workers", type=int, default=25, help="Workers (default 25)")
    parser.add_argument("--no-resume", action="store_true", help="Re-download all")

    args = parser.parse_args()
    scraper = UltimateFastPCFScraper()

    if args.days:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
        scraper.scrape_range_ultimate(start, end, args.workers, not args.no_resume)
    else:
        start = args.start or "2021-04-16"
        scraper.scrape_range_ultimate(start, args.end, args.workers, not args.no_resume)

if __name__ == "__main__":
    main()
