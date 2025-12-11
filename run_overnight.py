#!/usr/bin/env python3
"""
RenTech-Style Overnight Paper Trading Runner
=============================================

Run this on Hostinger VPS for overnight testing:

    nohup python run_overnight.py --hours 12 > overnight.log 2>&1 &

Or with screen:
    screen -S trading
    python run_overnight.py --hours 12
    # Ctrl+A, D to detach

Check logs:
    tail -f overnight.log
"""
import asyncio
import argparse
import logging
import time
from datetime import datetime, timedelta

from trading.rentech_engine import RenTechEngine


async def run_overnight(hours: float, capital: float):
    """Run RenTech engine for specified hours."""

    engine = RenTechEngine(
        capital_usd=capital,
        paper_mode=True,
        min_edge=0.002,          # 0.2% min edge after friction
        max_positions=5,          # More positions for more trades
        scan_interval_ms=5000,    # 5 second scans
        target_profit_pct=0.05,   # 5% target
        max_loss_pct=0.03,        # 3% stop
        max_hold_ms=60000,        # 60 second max hold
    )

    duration_seconds = int(hours * 3600)
    end_time = time.time() + duration_seconds

    print("=" * 70)
    print("  RENTECH OVERNIGHT PAPER TRADING")
    print("=" * 70)
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: {hours} hours ({duration_seconds} seconds)")
    print(f"  End: {(datetime.now() + timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Capital: ${capital:.2f}")
    print()
    print("  CONFIG:")
    print(f"    Min Edge: {engine.min_edge:.2%}")
    print(f"    Max Positions: {engine.max_positions}")
    print(f"    Target/Stop: {engine.target_profit_pct:.0%}/{engine.max_loss_pct:.0%}")
    print(f"    Scan Interval: {engine.scan_interval_ms}ms")
    print("=" * 70)
    print()

    try:
        await engine.start()

        # Status update every 5 minutes
        last_status = time.time()
        status_interval = 300  # 5 minutes

        while time.time() < end_time:
            await asyncio.sleep(30)

            # Periodic status update
            if time.time() - last_status >= status_interval:
                stats = engine.get_stats()
                elapsed_hours = (time.time() - (end_time - duration_seconds)) / 3600
                remaining_hours = (end_time - time.time()) / 3600

                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STATUS UPDATE")
                print(f"  Elapsed: {elapsed_hours:.1f}h | Remaining: {remaining_hours:.1f}h")
                print(f"  Trades: {stats['trades']} | W/L: {stats['wins']}/{stats['losses']}")
                print(f"  Win Rate: {stats['win_rate']:.1%}")
                print(f"  PnL: ${stats['total_pnl_usd']:+.2f} ({stats['total_pnl_pct']*100:+.1f}%)")
                print(f"  Capital: ${stats['capital_usd']:.2f}")
                print(f"  Open Positions: {stats['open_positions']}")
                print(f"  Scans: {stats['scans']} | Opportunities: {stats['opportunities']}")

                last_status = time.time()

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Stopping engine...")
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
    finally:
        await engine.stop()

        # Final report
        stats = engine.get_stats()
        print()
        print("=" * 70)
        print("  FINAL OVERNIGHT RESULTS")
        print("=" * 70)
        print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print(f"  Total Trades: {stats['trades']}")
        print(f"  Wins: {stats['wins']} | Losses: {stats['losses']}")
        print(f"  Win Rate: {stats['win_rate']:.2%}")
        print()
        print(f"  Starting Capital: ${stats['initial_capital']:.2f}")
        print(f"  Final Capital: ${stats['capital_usd']:.2f}")
        print(f"  Total PnL: ${stats['total_pnl_usd']:+.2f} ({stats['total_pnl_pct']*100:+.2f}%)")
        print()
        print(f"  Total Scans: {stats['scans']}")
        print(f"  Tokens Scanned: {stats['tokens_scanned']}")
        print(f"  Opportunities Found: {stats['opportunities']}")
        print()

        # RenTech comparison
        print("  RENTECH BENCHMARK COMPARISON:")
        print(f"    Target Win Rate: 50.75%")
        print(f"    Your Win Rate: {stats['win_rate']:.2%}")

        if stats['trades'] >= 100:
            print(f"    ✓ Good sample size ({stats['trades']} trades)")
        else:
            print(f"    ⚠ Need more trades for statistical significance ({stats['trades']}/100)")

        if stats['win_rate'] > 0.5:
            print(f"    ✓ Positive edge detected!")
        else:
            print(f"    ⚠ Edge needs improvement")

        print("=" * 70)

        # Save results to file
        results_file = f"overnight_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(results_file, 'w') as f:
            f.write(f"RenTech Overnight Trading Results\n")
            f.write(f"================================\n\n")
            f.write(f"Duration: {hours} hours\n")
            f.write(f"Capital: ${capital:.2f}\n\n")
            f.write(f"Results:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
        print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="RenTech-Style Overnight Paper Trading"
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=12.0,
        help="Duration to run (default: 12 hours)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10.0,
        help="Starting capital in USD (default: $10)"
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Run
    asyncio.run(run_overnight(args.hours, args.capital))


if __name__ == "__main__":
    main()
