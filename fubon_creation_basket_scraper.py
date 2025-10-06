#!/usr/bin/env python3
"""
Fubon ETF Ultimate Fast Scraper
Maximum speed + timeout bypass + intelligent retry

Optimizations:
- Aggressive parallel processing (20+ workers)
- Smart timeout handling (fast fail, quick retry)
- Connection pooling and session reuse
- Minimal delays between requests
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class UltimateFastScraper:
    def __init__(self, etf_code="00885", output_dir="/data/fubon_weight_data/creation_basket"):
        self.etf_code = etf_code
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.file_lock = threading.Lock()
        self.seen_data_dates = {}
        self.seen_lock = threading.Lock()
        
        # Disable SSL warnings
        import urllib3
        urllib3.disable_warnings()
        
        # Pre-create session pool for maximum speed
        self.session_pool = []
        for _ in range(30):  # Create 30 sessions for parallel use
            self.session_pool.append(self._create_fast_session())
        
        logger.info(f"🚀 Ultimate Fast Scraper initialized for ETF {etf_code}")
    
    def _create_fast_session(self):
        """Create optimized session for speed"""
        session = requests.Session()
        session.verify = False
        
        # Minimal headers for speed
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Aggressive retry with minimal backoff
        retry = Retry(
            total=2,  # Only 2 retries for speed
            backoff_factor=0.1,  # Minimal backoff
            status_forcelist=[500, 502, 503, 504]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=50,
            pool_maxsize=50
        )
        
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    def _extract_data_fast(self, html):
        """Fast extraction without error checking"""
        soup = BeautifulSoup(html, 'lxml')  # lxml is faster than html.parser
        
        # Fast extraction
        holdings = []
        total_info = {}
        tables = soup.find_all('table', class_='table1')
        
        for table in tables:
            if 'VIC VN' in str(table):
                for row in table.find_all('tr')[1:]:
                    cells = row.find_all('td')
                    if len(cells) >= 5:
                        stock_code = cells[0].text.strip()
                        if 'Total' in stock_code:
                            # Extract total row information
                            total_info = {
                                'total_market_value': cells[3].text.strip().replace(',', ''),
                                'total_weight': cells[4].text.strip().replace('%', '')
                            }
                            break
                        if stock_code and 'VN' in stock_code:
                            holdings.append({
                                'stock_code': stock_code,
                                'stock_name': cells[1].text.strip(),
                                'shares': cells[2].text.strip().replace(',', ''),
                                'market_value': cells[3].text.strip().replace(',', ''),
                                'weight_pct': cells[4].text.strip().replace('%', '')
                            })
                break
        
        # Complete NAV extraction
        nav_info = {}
        text = soup.get_text()
        
        # Extract all NAV fields
        nav_value_match = re.search(r'Net Asset Value[^\d]+([\d,]+)', text)
        if nav_value_match:
            nav_info['net_asset_value'] = nav_value_match.group(1).replace(',', '')
        
        units_match = re.search(r'Total Units Outstanding[^\d]+([\d,]+)', text)
        if units_match:
            nav_info['total_units'] = units_match.group(1).replace(',', '')
        
        nav_per_unit_match = re.search(r'NAV Per Unit[^\d]+([\d.]+)', text)
        if nav_per_unit_match:
            nav_info['nav_per_unit'] = nav_per_unit_match.group(1)
        
        # Fast exchange rate extraction
        exchange_info = {}
        usd_twd_match = re.search(r'1 USD\s*=\s*([\d.]+)\s*TWD', text)
        if usd_twd_match:
            exchange_info['usd_to_twd'] = float(usd_twd_match.group(1))
        
        usd_vnd_match = re.search(r'1 USD\s*=\s*([\d.,]+)\s*VND', text)
        if usd_vnd_match:
            # Handle VND which may have commas
            vnd_str = usd_vnd_match.group(1).replace(',', '')
            exchange_info['usd_to_vnd'] = float(vnd_str)
        
        # Extract cash holdings table
        cash_info = {}
        for table in tables:
            if 'Cash' in str(table) and 'Holdings' in str(table) and 'VIC VN' not in str(table):
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header row
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        label = cells[0].text.strip()
                        value = cells[1].text.strip().replace(',', '')
                        
                        # Map the cash holdings to field names
                        if 'Cash (TWD)' in label:
                            cash_info['cash_twd'] = value
                        elif 'Cash (USD)' in label:
                            cash_info['cash_usd'] = value
                        elif 'Cash (VND)' in label:
                            cash_info['cash_vnd'] = value
                        elif 'Payables' in label and 'TWD' in label:
                            cash_info['payables_twd'] = value
                break
        
        # Fast date extraction
        data_date = None
        date_match = re.search(r'Date:\s*(\d{4}/\d{2}/\d{2})', text)
        if date_match:
            data_date = date_match.group(1)
        
        return holdings, nav_info, exchange_info, total_info, cash_info, data_date
    
    def scrape_date_ultra_fast(self, date_obj, session_idx=0, skip_duplicates=True):
        """Ultra fast single date scraping with timeout bypass"""
        date_str = date_obj.strftime('%Y%m%d')
        date_display = date_obj.strftime('%Y-%m-%d')
        
        # Skip weekends
        if skip_duplicates and date_obj.weekday() >= 5:
            return (date_display, 'skipped', 'weekend')
        
        url = f"https://websys.fsit.com.tw/FubonETF/Trade/Assets.aspx?stkId={self.etf_code}&ddate={date_str}&lan=EN"
        
        # Get session from pool
        session = self.session_pool[session_idx % len(self.session_pool)]
        
        # Try with different timeouts (fast fail strategy)
        timeouts = [3, 5, 10]  # Start with very short timeout
        
        for attempt, timeout in enumerate(timeouts):
            try:
                response = session.get(url, timeout=timeout)
                
                if response.status_code == 200:
                    # Fast extraction
                    holdings, nav_info, exchange_info, total_info, cash_info, data_date = self._extract_data_fast(response.text)
                    
                    if not holdings:
                        if attempt < len(timeouts) - 1:
                            continue
                        return (date_display, 'failed', 'No data')
                    
                    if not data_date:
                        data_date = date_display
                    
                    # Check duplicates
                    with self.seen_lock:
                        if skip_duplicates and data_date in self.seen_data_dates:
                            return (date_display, 'duplicate', data_date)
                        self.seen_data_dates[data_date] = date_display
                    
                    # Fast DataFrame creation
                    df = pd.DataFrame(holdings)
                    df['requested_date'] = date_display
                    df['data_date'] = data_date
                    df['etf_code'] = self.etf_code
                    
                    # Add NAV info
                    for key, value in nav_info.items():
                        df[key] = value
                    
                    # Add exchange rates
                    if 'usd_to_twd' in exchange_info:
                        df['usd_to_twd'] = exchange_info['usd_to_twd']
                    if 'usd_to_vnd' in exchange_info:
                        df['usd_to_vnd'] = exchange_info['usd_to_vnd']
                    
                    # Add total info
                    if total_info:
                        df['total_market_value'] = total_info.get('total_market_value')
                        df['total_weight'] = total_info.get('total_weight')
                    
                    # Add cash holdings info
                    if cash_info:
                        for key, value in cash_info.items():
                            df[key] = value
                    
                    # Fast save
                    csv_file = os.path.join(self.output_dir, f"{self.etf_code}_{data_date.replace('/', '')}.csv")
                    with self.file_lock:
                        if not os.path.exists(csv_file):
                            df.to_csv(csv_file, index=False)
                    
                    return (date_display, 'success', {
                        'holdings': len(holdings),
                        'nav': nav_info.get('nav_per_unit', 'N/A'),
                        'timeout': timeout
                    })
                
                elif response.status_code == 404:
                    return (date_display, 'failed', '404')
                else:
                    if attempt < len(timeouts) - 1:
                        continue
                    return (date_display, 'failed', f'Status {response.status_code}')
                    
            except requests.exceptions.Timeout:
                if attempt < len(timeouts) - 1:
                    continue  # Try next timeout
                return (date_display, 'timeout', f'All timeouts failed')
            
            except Exception as e:
                if attempt < len(timeouts) - 1:
                    continue
                return (date_display, 'error', str(e)[:50])
        
        return (date_display, 'failed', 'All attempts failed')
    
    def get_existing_dates(self):
        """Fast check for existing dates"""
        existing = set()
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                if f.startswith(f"{self.etf_code}_") and f.endswith('.csv'):
                    date_part = f[6:14]  # Extract YYYYMMDD
                    if len(date_part) == 8 and date_part.isdigit():
                        existing.add(f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}")
        return existing
    
    def scrape_range_ultimate(self, start_date="2021-04-16", end_date=None, max_workers=25, resume=True):
        """Ultimate fast scraping with maximum parallelism"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get dates to process
        existing = set()
        if resume:
            existing = self.get_existing_dates()
            if existing:
                logger.info(f"📁 Found {len(existing)} existing files (resume mode ON)")
        
        dates_to_process = []
        current = start
        while current <= end:
            if not (resume and current.strftime('%Y-%m-%d') in existing):
                dates_to_process.append(current)
            current += timedelta(days=1)
        
        total = len(dates_to_process)
        
        if total == 0:
            logger.info("✅ All dates already downloaded!")
            return {'successful': [], 'failed': []}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"⚡ ULTIMATE FAST SCRAPING")
        logger.info(f"{'='*60}")
        logger.info(f"📅 Range: {start_date} to {end_date}")
        logger.info(f"📊 Days to process: {total}")
        logger.info(f"🚀 Workers: {max_workers} (MAXIMUM SPEED)")
        logger.info(f"⏱️  Timeout strategy: 3s → 5s → 10s")
        logger.info(f"{'='*60}\n")
        
        successful = []
        failed = []
        duplicates = []
        
        start_time = time.time()
        
        # Process with maximum parallelism
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all at once for maximum speed
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
                    
                    if status == 'success':
                        successful.append(date_str)
                        print(f"[{completed}/{total}] {date_str} ✓ {result['holdings']} stocks | NAV: {result['nav']} | {result['timeout']}s")
                    elif status == 'duplicate':
                        duplicates.append(date_str)
                        print(f"[{completed}/{total}] {date_str} ⚠ Dup")
                    elif status == 'skipped':
                        print(f"[{completed}/{total}] {date_str} - Skip")
                    elif status == 'timeout':
                        failed.append(date_str)
                        print(f"[{completed}/{total}] {date_str} ⏱️ Timeout")
                    else:
                        failed.append(date_str)
                        print(f"[{completed}/{total}] {date_str} ✗")
                except Exception as e:
                    failed.append(date.strftime('%Y-%m-%d'))
                    print(f"[{completed}/{total}] Error: {str(e)[:30]}")
        
        # Quick retry for failed dates with longer timeout
        if failed and len(failed) <= 20:  # Only retry if not too many failures
            logger.info(f"\n⚡ Quick retry for {len(failed)} failed dates...")
            retry_success = []
            
            for date_str in failed[:]:
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    result = self.scrape_date_ultra_fast(date_obj, 0, True)
                    if result[1] == 'success':
                        retry_success.append(date_str)
                        failed.remove(date_str)
                        successful.append(date_str)
                        print(f"  {date_str} ✓ Recovered")
                except:
                    pass
        
        elapsed = time.time() - start_time
        
        # Results
        logger.info(f"\n{'='*60}")
        logger.info(f"🏁 COMPLETE!")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Successful: {len(successful)} days")
        logger.info(f"⚠️  Duplicates: {len(duplicates)} days")
        logger.info(f"❌ Failed: {len(failed)} days")
        logger.info(f"⏱️  Total time: {elapsed:.1f}s")
        logger.info(f"⚡ Speed: {total/elapsed:.1f} days/second")
        logger.info(f"🚀 Avg per day: {elapsed/total*1000:.0f}ms")
        
        if failed:
            failed_file = os.path.join(self.output_dir, f"{self.etf_code}_failed.txt")
            with open(failed_file, 'w') as f:
                for d in failed:
                    f.write(f"{d}\n")
            logger.info(f"📝 Failed dates saved: {failed_file}")
        
        logger.info(f"{'='*60}\n")
        
        return {'successful': successful, 'failed': failed}

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🚀 ULTIMATE FAST Fubon ETF Scraper',
        epilog='Maximum speed with timeout bypass!'
    )
    
    parser.add_argument('--start', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', help='End date YYYY-MM-DD')
    parser.add_argument('--days', type=int, help='Last N days')
    parser.add_argument('--workers', type=int, default=25, help='Workers (default 25)')
    parser.add_argument('--no-resume', action='store_true', help='Re-download all')
    
    args = parser.parse_args()
    
    scraper = UltimateFastScraper()
    
    if args.days:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        scraper.scrape_range_ultimate(start, end, args.workers, not args.no_resume)
    else:
        start = args.start or '2021-04-16'
        scraper.scrape_range_ultimate(start, args.end, args.workers, not args.no_resume)

if __name__ == "__main__":
    main()