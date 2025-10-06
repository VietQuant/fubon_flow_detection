#!/usr/bin/env python3
"""
Add exchange rates using the EXACT same logic as fubon_scraper_ultimate_fast.py
"""

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
warnings.filterwarnings('ignore')

class ExchangeRateAdder:
    def __init__(self, data_dir="/data/fubon_weight_data/raw"):
        self.data_dir = data_dir
        self.etf_code = "00885"
        
        # Disable SSL warnings
        import urllib3
        urllib3.disable_warnings()
        
        # Pre-create session pool for maximum speed (FROM fubon_scraper_ultimate_fast.py)
        self.session_pool = []
        for _ in range(30):  # Create 30 sessions for parallel use
            self.session_pool.append(self._create_fast_session())
    
    def _create_fast_session(self):
        """Create optimized session for speed (FROM fubon_scraper_ultimate_fast.py)"""
        session = requests.Session()
        session.verify = False
        
        # Minimal headers for speed
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        return session
    
    def fetch_exchange_rates_for_date(self, date_str, session_idx=0):
        """Fetch exchange rates using EXACT logic from fubon_scraper_ultimate_fast.py"""
        url = f"https://websys.fsit.com.tw/FubonETF/Trade/Assets.aspx?stkId={self.etf_code}&ddate={date_str}&lan=EN"
        
        # Get session from pool
        session = self.session_pool[session_idx % len(self.session_pool)]
        
        # Try with different timeouts (fast fail strategy FROM fubon_scraper_ultimate_fast.py)
        timeouts = [3, 5, 10]  # Start with very short timeout
        
        for attempt, timeout in enumerate(timeouts):
            try:
                response = session.get(url, timeout=timeout)
                
                if response.status_code == 200:
                    # Use lxml for speed (FROM fubon_scraper_ultimate_fast.py)
                    soup = BeautifulSoup(response.text, 'lxml')
                    text = soup.get_text()
                    
                    # Fast exchange rate extraction (EXACT PATTERN from fubon_scraper_ultimate_fast.py)
                    exchange_info = {}
                    usd_twd_match = re.search(r'1 USD\s*=\s*([\d.]+)\s*TWD', text)
                    if usd_twd_match:
                        exchange_info['usd_to_twd'] = float(usd_twd_match.group(1))
                    
                    usd_vnd_match = re.search(r'1 USD\s*=\s*([\d.]+)\s*VND', text)
                    if usd_vnd_match:
                        exchange_info['usd_to_vnd'] = float(usd_vnd_match.group(1))
                    
                    if exchange_info:
                        return date_str, exchange_info, timeout
                    
            except requests.exceptions.Timeout:
                if attempt < len(timeouts) - 1:
                    continue  # Try next timeout
            except:
                if attempt < len(timeouts) - 1:
                    continue
        
        return date_str, {}, None
    
    def run(self):
        """Main execution with parallel processing (FROM fubon_scraper_ultimate_fast.py style)"""
        files = sorted(glob(os.path.join(self.data_dir, f"{self.etf_code}_*.csv")))
        
        print("🔍 Scanning files for missing exchange rates...")
        
        files_needing_rates = []
        for file in files:
            df = pd.read_csv(file, nrows=1)
            if 'usd_to_twd' not in df.columns:
                files_needing_rates.append(file)
        
        if not files_needing_rates:
            print("\n✅ All files already have exchange rates!")
            return
        
        # Extract unique dates
        dates_to_fetch = set()
        for file in files_needing_rates:
            filename = os.path.basename(file)
            date_str = filename.replace(f"{self.etf_code}_", "").replace(".csv", "")
            dates_to_fetch.add(date_str)
        
        dates_to_fetch = sorted(list(dates_to_fetch))
        
        print(f"\n{'='*60}")
        print(f"⚡ ADDING EXCHANGE RATES (ULTIMATE FAST)")
        print(f"{'='*60}")
        print(f"📁 Total files: {len(files)}")
        print(f"📝 Files needing rates: {len(files_needing_rates)}")
        print(f"📊 Unique dates to fetch: {len(dates_to_fetch)}")
        print(f"🚀 Workers: 25 (MAXIMUM SPEED)")
        print(f"⏱️  Timeout strategy: 3s → 5s → 10s")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        exchange_rates = {}
        
        # Process with maximum parallelism (FROM fubon_scraper_ultimate_fast.py)
        with ThreadPoolExecutor(max_workers=25) as executor:
            # Submit all at once for maximum speed
            futures = {
                executor.submit(self.fetch_exchange_rates_for_date, date, idx): (date, idx)
                for idx, date in enumerate(dates_to_fetch)
            }
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                date, idx = futures[future]
                
                try:
                    date_str, rates, timeout_used = future.result()
                    
                    # Format date for display
                    if len(date_str) == 8:
                        display_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                    else:
                        display_date = date_str
                    
                    if rates:
                        exchange_rates[date_str] = rates
                        twd = rates.get('usd_to_twd', 'N/A')
                        vnd = rates.get('usd_to_vnd', 'N/A')
                        print(f"[{completed}/{len(dates_to_fetch)}] {display_date} ✓ TWD: {twd} | VND: {vnd} | {timeout_used}s")
                    else:
                        print(f"[{completed}/{len(dates_to_fetch)}] {display_date} ✗")
                except Exception as e:
                    print(f"[{completed}/{len(dates_to_fetch)}] Error: {str(e)[:30]}")
        
        fetch_time = time.time() - start_time
        
        # Update files
        print(f"\n📝 Updating {len(files_needing_rates)} CSV files...")
        updated = 0
        
        for file in files_needing_rates:
            filename = os.path.basename(file)
            date_str = filename.replace(f"{self.etf_code}_", "").replace(".csv", "")
            
            if date_str in exchange_rates:
                rates = exchange_rates[date_str]
                
                # Read and update file
                df = pd.read_csv(file)
                df['usd_to_twd'] = rates.get('usd_to_twd')
                df['usd_to_vnd'] = rates.get('usd_to_vnd')
                
                # Save
                df.to_csv(file, index=False)
                updated += 1
                
                if updated % 100 == 0:
                    print(f"   Progress: {updated}/{len(files_needing_rates)}")
        
        total_time = time.time() - start_time
        
        # Results (EXACT style from fubon_scraper_ultimate_fast.py)
        print(f"\n{'='*60}")
        print(f"🏁 COMPLETE!")
        print(f"{'='*60}")
        print(f"✅ Files updated: {updated}")
        print(f"❌ Files not updated: {len(files_needing_rates) - updated}")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"⚡ Fetch speed: {len(dates_to_fetch)/fetch_time:.1f} dates/second")
        print(f"🚀 Avg per date: {fetch_time/len(dates_to_fetch)*1000:.0f}ms")
        print(f"{'='*60}\n")
        
        # Verify
        if updated > 0:
            sample_file = files_needing_rates[0]
            df = pd.read_csv(sample_file)
            print(f"🔍 Verifying: {os.path.basename(sample_file)}")
            if 'usd_to_twd' in df.columns:
                print(f"   ✅ USD to TWD: {df['usd_to_twd'].iloc[0]}")
                print(f"   ✅ USD to VND: {df['usd_to_vnd'].iloc[0]}")

def main():
    adder = ExchangeRateAdder()
    adder.run()

if __name__ == "__main__":
    main()