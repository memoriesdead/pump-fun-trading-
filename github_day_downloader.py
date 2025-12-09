#!/usr/bin/env python3
"""
GitHub Actions Day Downloader
=============================
Downloads a single day of pump.fun trades from CryptoHouse.
Each GitHub Actions runner = different IP = bypasses rate limits.

Usage: python github_day_downloader.py <day_number>
  day_number: 0-281 (0 = 2024-03-01, 281 = 2024-12-07)
"""
import requests
import gzip
import json
import struct
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path('data')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CRYPTOHOUSE_URL = 'https://crypto-clickhouse.clickhouse.com:8443'
PUMP_PROGRAM = '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'

BUY_DISC = bytes.fromhex('66063d1201daebea')
SELL_DISC = bytes.fromhex('33e685a4017f83ad')

B58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
B58_MAP = {c: i for i, c in enumerate(B58_ALPHABET)}

def b58decode(s):
    num = 0
    for c in s:
        if c not in B58_MAP:
            return b''
        num = num * 58 + B58_MAP[c]
    result = []
    while num:
        result.append(num % 256)
        num //= 256
    return bytes(reversed(result)) if result else b''

def day_to_date(day_num):
    """Convert day number to date string (0 = 2024-03-01)"""
    base = datetime(2024, 3, 1)
    target = base + timedelta(days=day_num)
    return target.strftime('%Y-%m-%d')

def download_day(date_str, session):
    """Download all trades for a single day"""
    out_file = OUTPUT_DIR / f'gh_{date_str}.jsonl.gz'

    start_time = f'{date_str} 00:00:00'
    next_day = (datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    end_time = f'{next_day} 00:00:00'

    all_trades = []
    offset = 0

    print(f"[*] Downloading {date_str}...")

    while offset < 5000000:  # Safety limit
        query = f"""SELECT tx_signature, block_slot, toString(block_timestamp) as ts, data
FROM solana.instructions WHERE program_id = '{PUMP_PROGRAM}'
AND block_timestamp >= toDateTime64('{start_time}', 6)
AND block_timestamp < toDateTime64('{end_time}', 6)
ORDER BY block_slot LIMIT 100000 OFFSET {offset} FORMAT JSONEachRow"""

        for attempt in range(5):
            try:
                r = session.post(CRYPTOHOUSE_URL, data=query, auth=('crypto', ''), timeout=180)

                if 'QUOTA_EXCEEDED' in r.text or 'TOO_MANY' in r.text:
                    print(f"  [!] Quota hit, waiting 60s...")
                    time.sleep(60)
                    continue

                if 'TIMEOUT' in r.text:
                    print(f"  [!] Timeout, retrying...")
                    time.sleep(2)
                    continue

                lines = [l for l in r.text.strip().split('\n') if l.strip() and l.startswith('{')]

                if not lines:
                    print(f"  [*] No more data at offset {offset}")
                    break

                batch_trades = 0
                for line in lines:
                    try:
                        row = json.loads(line)
                        data = b58decode(row.get('data', ''))
                        if len(data) < 24:
                            continue

                        disc = data[:8]

                        if disc == BUY_DISC:
                            tokens = struct.unpack('<Q', data[8:16])[0] / 1e6
                            sol = struct.unpack('<Q', data[16:24])[0] / 1e9
                            all_trades.append({
                                'sig': row['tx_signature'],
                                'slot': row['block_slot'],
                                'ts': row['ts'],
                                'type': 'buy',
                                'tokens': round(tokens, 6),
                                'sol': round(sol, 9)
                            })
                            batch_trades += 1

                        elif disc == SELL_DISC:
                            tokens = struct.unpack('<Q', data[8:16])[0] / 1e6
                            sol = struct.unpack('<Q', data[16:24])[0] / 1e9
                            all_trades.append({
                                'sig': row['tx_signature'],
                                'slot': row['block_slot'],
                                'ts': row['ts'],
                                'type': 'sell',
                                'tokens': round(tokens, 6),
                                'sol': round(sol, 9)
                            })
                            batch_trades += 1
                    except:
                        pass

                print(f"  [+] Offset {offset}: {len(lines)} rows, {batch_trades} trades")
                offset += 100000

                if len(lines) < 100000:
                    break

                break  # Success, exit retry loop

            except Exception as e:
                print(f"  [!] Error: {e}")
                if attempt == 4:
                    break
                time.sleep(1)
        else:
            break  # All retries failed

    # Save results
    if all_trades:
        with gzip.open(out_file, 'wt') as f:
            for t in all_trades:
                f.write(json.dumps(t) + '\n')
        print(f"[+] Saved {len(all_trades):,} trades to {out_file.name}")
    else:
        print(f"[-] No trades found for {date_str}")

    return len(all_trades)

def main():
    if len(sys.argv) < 2:
        print("Usage: python github_day_downloader.py <day_number>")
        print("  day_number: 0-281 (0 = 2024-03-01, 281 = 2024-12-07)")
        sys.exit(1)

    day_num = int(sys.argv[1])

    if day_num < 0 or day_num > 281:
        print(f"Error: day_number must be 0-281, got {day_num}")
        sys.exit(1)

    date_str = day_to_date(day_num)
    print(f"=" * 60)
    print(f"  GitHub Actions PumpFun Downloader")
    print(f"  Day {day_num} = {date_str}")
    print(f"=" * 60)

    session = requests.Session()
    trades = download_day(date_str, session)

    print(f"\n{'=' * 60}")
    print(f"  Complete! {trades:,} trades downloaded")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
