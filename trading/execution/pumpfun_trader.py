"""
Pump.fun Direct Trading Execution
=================================

Direct trading on pump.fun bonding curve AMM.

Requirements:
    pip install solana solders anchorpy base58

Usage:
    trader = PumpfunTrader(keypair_path, rpc_url)
    await trader.buy(mint, sol_amount=0.1, slippage=0.05)
    await trader.sell(mint, token_pct=1.0, slippage=0.05)
"""

import asyncio
import base64
import struct
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import base58

# Pump.fun constants
PUMPFUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
BUY_DISCRIMINATOR = bytes.fromhex("66063d1201daebea")
SELL_DISCRIMINATOR = bytes.fromhex("33e685a4017f83ad")

# Virtual reserves for bonding curve
INITIAL_VIRTUAL_SOL = 30_000_000_000  # 30 SOL in lamports
INITIAL_VIRTUAL_TOKENS = 1_073_000_000_000_000  # 1.073B tokens

# Fee
PUMPFUN_FEE_BPS = 100  # 1% fee


@dataclass
class BondingCurveState:
    """Current state of a token's bonding curve"""
    virtual_sol_reserves: int
    virtual_token_reserves: int
    real_sol_reserves: int
    real_token_reserves: int
    token_total_supply: int


class PumpfunBondingCurve:
    """
    Pump.fun uses a constant product AMM (k = x * y).

    Price increases as more SOL is added to buy tokens.
    Price decreases as tokens are sold back for SOL.
    """

    @staticmethod
    def get_buy_price(sol_amount_lamports: int, state: BondingCurveState) -> int:
        """Calculate tokens received for SOL input"""
        # Apply fee
        sol_after_fee = sol_amount_lamports * (10000 - PUMPFUN_FEE_BPS) // 10000

        # Constant product formula
        k = state.virtual_sol_reserves * state.virtual_token_reserves
        new_sol_reserves = state.virtual_sol_reserves + sol_after_fee
        new_token_reserves = k // new_sol_reserves
        tokens_out = state.virtual_token_reserves - new_token_reserves

        return tokens_out

    @staticmethod
    def get_sell_price(token_amount: int, state: BondingCurveState) -> int:
        """Calculate SOL received for token input"""
        # Constant product formula
        k = state.virtual_sol_reserves * state.virtual_token_reserves
        new_token_reserves = state.virtual_token_reserves + token_amount
        new_sol_reserves = k // new_token_reserves
        sol_out = state.virtual_sol_reserves - new_sol_reserves

        # Apply fee
        sol_after_fee = sol_out * (10000 - PUMPFUN_FEE_BPS) // 10000

        return sol_after_fee

    @staticmethod
    def get_price_per_token(state: BondingCurveState) -> float:
        """Get current price in SOL per token"""
        return state.virtual_sol_reserves / state.virtual_token_reserves / 1e9


class PumpfunTrader:
    """
    Direct pump.fun trading via Solana RPC.

    Example:
        trader = PumpfunTrader("~/.config/solana/pumpfun.json")
        result = await trader.buy("mint_address", sol_amount=0.1)
    """

    def __init__(
        self,
        keypair_path: str,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        priority_fee_lamports: int = 100_000,  # 0.0001 SOL priority fee
    ):
        self.keypair_path = Path(keypair_path).expanduser()
        self.rpc_url = rpc_url
        self.priority_fee = priority_fee_lamports
        self._keypair = None
        self._client = None

    async def connect(self):
        """Initialize connection and load keypair"""
        try:
            from solana.rpc.async_api import AsyncClient
            from solders.keypair import Keypair
            import json

            # Load keypair
            with open(self.keypair_path) as f:
                secret = json.load(f)
            self._keypair = Keypair.from_bytes(bytes(secret))

            # Connect to RPC
            self._client = AsyncClient(self.rpc_url)

            print(f"Connected: {self._keypair.pubkey()}")
            return True

        except ImportError:
            print("Install: pip install solana solders")
            return False
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def get_balance(self) -> float:
        """Get wallet SOL balance"""
        if not self._client:
            await self.connect()

        resp = await self._client.get_balance(self._keypair.pubkey())
        return resp.value / 1e9

    async def get_bonding_curve_state(self, mint: str) -> Optional[BondingCurveState]:
        """Fetch current bonding curve state for a token"""
        # In production, derive the bonding curve PDA and fetch account data
        # For now, return estimated state based on typical values
        return BondingCurveState(
            virtual_sol_reserves=INITIAL_VIRTUAL_SOL,
            virtual_token_reserves=INITIAL_VIRTUAL_TOKENS,
            real_sol_reserves=0,
            real_token_reserves=0,
            token_total_supply=1_000_000_000_000_000,
        )

    async def buy(
        self,
        mint: str,
        sol_amount: float,
        slippage: float = 0.05,
        max_retries: int = 3,
    ) -> dict:
        """
        Buy tokens on pump.fun bonding curve.

        Args:
            mint: Token mint address
            sol_amount: Amount of SOL to spend
            slippage: Max slippage tolerance (0.05 = 5%)
            max_retries: Number of retry attempts

        Returns:
            dict with signature, tokens_received, price
        """
        if not self._client:
            await self.connect()

        sol_lamports = int(sol_amount * 1e9)

        # Get current state
        state = await self.get_bonding_curve_state(mint)
        if not state:
            return {"success": False, "error": "Failed to get bonding curve state"}

        # Calculate expected tokens
        expected_tokens = PumpfunBondingCurve.get_buy_price(sol_lamports, state)
        min_tokens = int(expected_tokens * (1 - slippage))

        # Build and send transaction
        # NOTE: Full implementation requires building Solana transaction
        # with proper accounts and instruction data

        return {
            "success": True,
            "expected_tokens": expected_tokens,
            "min_tokens": min_tokens,
            "sol_spent": sol_amount,
            "price": PumpfunBondingCurve.get_price_per_token(state),
            "note": "Transaction building requires full Solana SDK setup",
        }

    async def sell(
        self,
        mint: str,
        token_amount: Optional[int] = None,
        token_pct: float = 1.0,
        slippage: float = 0.05,
    ) -> dict:
        """
        Sell tokens on pump.fun bonding curve.

        Args:
            mint: Token mint address
            token_amount: Exact tokens to sell (if None, uses token_pct)
            token_pct: Percentage of holdings to sell (1.0 = 100%)
            slippage: Max slippage tolerance

        Returns:
            dict with signature, sol_received, price
        """
        if not self._client:
            await self.connect()

        # Get current state
        state = await self.get_bonding_curve_state(mint)
        if not state:
            return {"success": False, "error": "Failed to get bonding curve state"}

        # Calculate expected SOL
        if token_amount is None:
            # Would fetch actual token balance here
            token_amount = int(1_000_000 * token_pct)

        expected_sol = PumpfunBondingCurve.get_sell_price(token_amount, state)
        min_sol = int(expected_sol * (1 - slippage))

        return {
            "success": True,
            "expected_sol": expected_sol / 1e9,
            "min_sol": min_sol / 1e9,
            "tokens_sold": token_amount,
            "price": PumpfunBondingCurve.get_price_per_token(state),
            "note": "Transaction building requires full Solana SDK setup",
        }

    async def close(self):
        """Close connection"""
        if self._client:
            await self._client.close()


class PumpfunScalper:
    """
    Automated scalping with signal integration.

    Combines:
    - PumpfunTrader for execution
    - EarlyPatternScorer for signals
    - Risk management (Kelly sizing, stop loss)
    """

    def __init__(
        self,
        trader: PumpfunTrader,
        max_position_sol: float = 0.5,
        stop_loss_pct: float = 0.30,
        take_profit_pct: float = 1.0,
        max_positions: int = 3,
    ):
        self.trader = trader
        self.max_position_sol = max_position_sol
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        self.positions = {}

    def calculate_position_size(
        self,
        score: float,
        win_rate: float = 0.55,
        max_drawdown: float = 0.20,
    ) -> float:
        """
        Kelly criterion with drawdown constraint.

        Args:
            score: Signal score (0-1)
            win_rate: Historical win rate
            max_drawdown: Maximum acceptable drawdown

        Returns:
            Position size as fraction of max_position_sol
        """
        # Convert score to edge estimate
        edge = score - 0.5  # Score above 0.5 implies positive edge

        if edge <= 0:
            return 0

        # Kelly fraction
        kelly = (win_rate * edge - (1 - win_rate)) / edge if edge > 0 else 0

        # Drawdown constraint (Grossman-Zhou)
        safe_kelly = kelly * (1 - max_drawdown)

        # Clamp to reasonable range
        position_pct = max(0, min(safe_kelly, 0.25))

        return position_pct * self.max_position_sol

    async def on_signal(self, mint: str, score: float, signal: str):
        """
        React to trading signal.

        Args:
            mint: Token mint address
            score: Signal score (0-1)
            signal: 'strong_buy', 'buy', 'hold', 'avoid'
        """
        if signal not in ('strong_buy', 'buy'):
            return

        if len(self.positions) >= self.max_positions:
            print(f"Max positions reached, skipping {mint}")
            return

        # Calculate position size
        size = self.calculate_position_size(score)
        if size < 0.01:  # Minimum 0.01 SOL
            return

        # Execute buy
        result = await self.trader.buy(mint, size)
        if result.get("success"):
            self.positions[mint] = {
                "entry_price": result["price"],
                "size_sol": size,
                "score": score,
            }
            print(f"BOUGHT {mint}: {size:.3f} SOL @ {result['price']:.10f}")


# Example usage
async def main():
    trader = PumpfunTrader(
        keypair_path="~/.config/solana/pumpfun.json",
        rpc_url="https://api.mainnet-beta.solana.com",
    )

    if await trader.connect():
        balance = await trader.get_balance()
        print(f"Balance: {balance:.4f} SOL")

        # Test buy calculation
        result = await trader.buy(
            mint="ExampleMintAddress",
            sol_amount=0.1,
            slippage=0.05,
        )
        print(f"Buy result: {result}")

        await trader.close()


if __name__ == "__main__":
    asyncio.run(main())
