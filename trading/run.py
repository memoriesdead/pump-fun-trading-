#!/usr/bin/env python3
"""
Trading System Runner
=====================

Safe entry point for the trading system with pre-flight checks.

Usage:
    # Paper trading (safe, no real money)
    python -m trading.run --mode paper --capital 100

    # Dry run (real executor, but no actual transactions)
    python -m trading.run --mode real --keypair ~/.config/solana/trading.json --dry-run

    # Real trading (LIVE MONEY - requires confirmation)
    python -m trading.run --mode real --keypair ~/.config/solana/trading.json --confirm-real

RenTech Principle: Paper and Real use IDENTICAL logic.
The ONLY difference is the executor.
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================

def check_python_version():
    """Ensure Python 3.9+"""
    if sys.version_info < (3, 9):
        logger.error(f"Python 3.9+ required, got {sys.version}")
        return False
    return True


def check_dependencies():
    """Check required packages are installed"""
    required = [
        'aiohttp',
        'numpy',
    ]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error("Install with: pip install " + ' '.join(missing))
        return False
    return True


def check_keypair(keypair_path: str) -> bool:
    """Verify keypair file exists and is valid"""
    path = Path(keypair_path).expanduser()

    if not path.exists():
        logger.error(f"Keypair file not found: {path}")
        return False

    try:
        with open(path) as f:
            data = json.load(f)
            if not isinstance(data, list) or len(data) != 64:
                logger.error("Invalid keypair format (expected 64-byte array)")
                return False
    except json.JSONDecodeError:
        logger.error("Keypair file is not valid JSON")
        return False
    except Exception as e:
        logger.error(f"Could not read keypair: {e}")
        return False

    logger.info(f"Keypair valid: {path}")
    return True


def check_rpc_connection(rpc_url: str) -> bool:
    """Test RPC endpoint connectivity"""
    import asyncio
    import aiohttp

    async def test_rpc():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    rpc_url,
                    json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    data = await resp.json()
                    if 'result' in data and data['result'] == 'ok':
                        return True
                    # Some RPCs return different format
                    if resp.status == 200:
                        return True
        except Exception as e:
            logger.error(f"RPC connection failed: {e}")
            return False
        return False

    return asyncio.run(test_rpc())


def check_websocket_connection() -> bool:
    """Test PumpPortal WebSocket connectivity"""
    import asyncio
    import aiohttp

    async def test_ws():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    "wss://pumpportal.fun/api/data",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as ws:
                    # Try to subscribe
                    await ws.send_json({"method": "subscribeNewToken"})
                    # Wait for any response
                    msg = await asyncio.wait_for(ws.receive(), timeout=5)
                    return msg.type == aiohttp.WSMsgType.TEXT
        except asyncio.TimeoutError:
            # Timeout is OK - connection worked
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    return asyncio.run(test_ws())


def display_capital_warning(capital: float, mode: str):
    """Display capital at risk warning for real mode"""
    if mode != 'real':
        return

    print("\n" + "=" * 60)
    print("  *** REAL MONEY WARNING ***")
    print("=" * 60)
    print(f"\n  Mode:     REAL (LIVE TRANSACTIONS)")
    print(f"  Capital:  {capital} SOL at risk")
    print(f"\n  This will execute REAL trades on Solana mainnet.")
    print("  Losses are possible and irreversible.")
    print("\n" + "=" * 60)


def confirm_real_trading() -> bool:
    """Require explicit confirmation for real trading"""
    print("\n  Type 'I ACCEPT THE RISK' to proceed: ", end='')
    try:
        response = input().strip()
        return response == 'I ACCEPT THE RISK'
    except EOFError:
        return False


# =============================================================================
# MAIN RUNNER
# =============================================================================

async def run_trading(args):
    """Run the trading system"""
    from .orchestrator import LiveTradingOrchestrator

    orchestrator = LiveTradingOrchestrator(
        paper_mode=(args.mode == 'paper'),
        capital=args.capital,
        keypair_path=args.keypair,
        rpc_url=args.rpc,
        dry_run=args.dry_run,
    )

    await orchestrator.start()


def main():
    parser = argparse.ArgumentParser(
        description='Trading System Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Paper trading:
    python -m trading.run --mode paper --capital 100

  Dry run (test real executor without transactions):
    python -m trading.run --mode real --keypair ~/.config/solana/trading.json --dry-run

  Real trading:
    python -m trading.run --mode real --keypair ~/.config/solana/trading.json --confirm-real
        """
    )

    parser.add_argument(
        '--mode',
        choices=['paper', 'real'],
        default='paper',
        help='Trading mode (default: paper)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100.0,
        help='Starting capital in SOL (default: 100)'
    )
    parser.add_argument(
        '--keypair',
        type=str,
        default=None,
        help='Path to Solana keypair JSON (required for real mode)'
    )
    parser.add_argument(
        '--rpc',
        type=str,
        default='https://api.mainnet-beta.solana.com',
        help='Solana RPC URL'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Real mode but simulate transactions (no actual sends)'
    )
    parser.add_argument(
        '--confirm-real',
        action='store_true',
        help='Confirm real money trading (required for real mode without --dry-run)'
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip pre-flight connectivity checks'
    )

    args = parser.parse_args()

    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    print("\n" + "=" * 60)
    print("  TRADING SYSTEM - PRE-FLIGHT CHECKS")
    print("=" * 60 + "\n")

    # Python version
    if not check_python_version():
        sys.exit(1)
    logger.info("[OK] Python version")

    # Dependencies
    if not check_dependencies():
        sys.exit(1)
    logger.info("[OK] Dependencies")

    # Mode-specific validation
    if args.mode == 'real':
        # Keypair required
        if not args.keypair:
            logger.error("--keypair required for real mode")
            sys.exit(1)

        if not check_keypair(args.keypair):
            sys.exit(1)
        logger.info("[OK] Keypair valid")

        # Must either be dry-run or explicitly confirmed
        if not args.dry_run and not args.confirm_real:
            logger.error("Real mode requires --dry-run OR --confirm-real")
            logger.error("Use --dry-run to test without sending transactions")
            logger.error("Use --confirm-real to enable live trading")
            sys.exit(1)

    # Connectivity checks
    if not args.skip_checks:
        logger.info("Checking connectivity...")

        if args.mode == 'real':
            if not check_rpc_connection(args.rpc):
                logger.error("RPC connection failed - cannot proceed")
                sys.exit(1)
            logger.info("[OK] Solana RPC connected")

        if not check_websocket_connection():
            logger.error("PumpPortal WebSocket failed - cannot proceed")
            sys.exit(1)
        logger.info("[OK] PumpPortal WebSocket connected")

    # ==========================================================================
    # CONFIRMATION FOR REAL TRADING
    # ==========================================================================

    if args.mode == 'real' and not args.dry_run:
        display_capital_warning(args.capital, args.mode)

        if not confirm_real_trading():
            logger.info("Real trading cancelled")
            sys.exit(0)

    # ==========================================================================
    # LAUNCH
    # ==========================================================================

    print("\n" + "=" * 60)
    print("  LAUNCHING TRADING SYSTEM")
    print("=" * 60)
    print(f"\n  Mode:     {'PAPER' if args.mode == 'paper' else 'REAL'}")
    print(f"  Capital:  {args.capital} SOL")
    if args.mode == 'real':
        print(f"  Dry Run:  {args.dry_run}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 60 + "\n")

    # Run
    try:
        asyncio.run(run_trading(args))
    except KeyboardInterrupt:
        print("\n\nShutdown requested...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
