"""
Solana Transaction Builder for Pump.fun
========================================

Full transaction building for pump.fun bonding curve trades.

Derived from pump.fun on-chain program analysis:
- Program: 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P
- Buy/Sell via constant product AMM

Requirements:
    pip install solana solders base58
"""

import struct
from typing import List, Tuple, Optional
from dataclasses import dataclass
import hashlib

import base58


# Pump.fun Program Constants
PUMPFUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
GLOBAL_STATE = "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf"
FEE_RECIPIENT = "CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbCJtDRxEy3nF9"
EVENT_AUTHORITY = "Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1"

# System programs
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
ASSOCIATED_TOKEN_PROGRAM = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
SYSTEM_PROGRAM = "11111111111111111111111111111111"
RENT_SYSVAR = "SysvarRent111111111111111111111111111111111"

# Discriminators (first 8 bytes of instruction)
BUY_DISCRIMINATOR = bytes.fromhex("66063d1201daebea")
SELL_DISCRIMINATOR = bytes.fromhex("33e685a4017f83ad")
CREATE_DISCRIMINATOR = bytes.fromhex("181ec828051c0777")


@dataclass
class AccountMeta:
    """Account metadata for Solana instructions"""
    pubkey: str
    is_signer: bool
    is_writable: bool


def derive_pda(seeds: List[bytes], program_id: str) -> Tuple[str, int]:
    """
    Derive Program Derived Address (PDA).

    Args:
        seeds: List of seed bytes
        program_id: Program ID as base58 string

    Returns:
        (pda_address, bump_seed)
    """
    program_id_bytes = base58.b58decode(program_id)

    for bump in range(255, -1, -1):
        try:
            seed_bytes = b"".join(seeds) + bytes([bump])
            hash_input = seed_bytes + program_id_bytes + b"ProgramDerivedAddress"
            hash_result = hashlib.sha256(hash_input).digest()

            # Check if valid ed25519 point (off-curve)
            # Simplified check - full implementation needs ed25519 validation
            if hash_result[31] < 0x80:
                return (base58.b58encode(hash_result).decode(), bump)
        except Exception:
            continue

    raise ValueError("Could not find valid PDA")


def derive_bonding_curve_pda(mint: str) -> str:
    """Derive bonding curve PDA for a token mint"""
    seeds = [
        b"bonding-curve",
        base58.b58decode(mint)
    ]
    pda, _ = derive_pda(seeds, PUMPFUN_PROGRAM_ID)
    return pda


def derive_associated_bonding_curve(mint: str, bonding_curve: str) -> str:
    """Derive associated token account for bonding curve"""
    seeds = [
        base58.b58decode(bonding_curve),
        base58.b58decode(TOKEN_PROGRAM_ID),
        base58.b58decode(mint)
    ]
    pda, _ = derive_pda(seeds, ASSOCIATED_TOKEN_PROGRAM)
    return pda


def derive_user_ata(user: str, mint: str) -> str:
    """Derive user's Associated Token Account"""
    seeds = [
        base58.b58decode(user),
        base58.b58decode(TOKEN_PROGRAM_ID),
        base58.b58decode(mint)
    ]
    pda, _ = derive_pda(seeds, ASSOCIATED_TOKEN_PROGRAM)
    return pda


class PumpfunInstructionBuilder:
    """
    Build pump.fun buy/sell instructions.

    Account layout for BUY (11 accounts):
        0: global (read)
        1: feeRecipient (write)
        2: mint (read)
        3: bondingCurve (write)
        4: associatedBondingCurve (write)
        5: associatedUser (write)
        6: user (signer, write)
        7: systemProgram (read)
        8: tokenProgram (read)
        9: rent (read)
        10: eventAuthority (read)
        11: program (read)

    Account layout for SELL (10 accounts):
        0: global (read)
        1: feeRecipient (write)
        2: mint (read)
        3: bondingCurve (write)
        4: associatedBondingCurve (write)
        5: associatedUser (write)
        6: user (signer, write)
        7: systemProgram (read)
        8: tokenProgram (read)
        9: eventAuthority (read)
        10: program (read)
    """

    @staticmethod
    def build_buy_instruction(
        user_pubkey: str,
        mint: str,
        token_amount: int,
        max_sol_cost: int,
    ) -> Tuple[bytes, List[AccountMeta]]:
        """
        Build BUY instruction data and accounts.

        Args:
            user_pubkey: User's wallet pubkey
            mint: Token mint address
            token_amount: Amount of tokens to buy
            max_sol_cost: Maximum SOL to spend (slippage protection)

        Returns:
            (instruction_data, account_metas)
        """
        # Derive PDAs
        bonding_curve = derive_bonding_curve_pda(mint)
        associated_bonding_curve = derive_associated_bonding_curve(mint, bonding_curve)
        user_ata = derive_user_ata(user_pubkey, mint)

        # Instruction data: discriminator + token_amount (u64) + max_sol_cost (u64)
        data = BUY_DISCRIMINATOR
        data += struct.pack("<Q", token_amount)  # u64 little-endian
        data += struct.pack("<Q", max_sol_cost)  # u64 little-endian

        # Account metas
        accounts = [
            AccountMeta(GLOBAL_STATE, False, False),
            AccountMeta(FEE_RECIPIENT, False, True),
            AccountMeta(mint, False, False),
            AccountMeta(bonding_curve, False, True),
            AccountMeta(associated_bonding_curve, False, True),
            AccountMeta(user_ata, False, True),
            AccountMeta(user_pubkey, True, True),
            AccountMeta(SYSTEM_PROGRAM, False, False),
            AccountMeta(TOKEN_PROGRAM_ID, False, False),
            AccountMeta(RENT_SYSVAR, False, False),
            AccountMeta(EVENT_AUTHORITY, False, False),
            AccountMeta(PUMPFUN_PROGRAM_ID, False, False),
        ]

        return (data, accounts)

    @staticmethod
    def build_sell_instruction(
        user_pubkey: str,
        mint: str,
        token_amount: int,
        min_sol_output: int,
    ) -> Tuple[bytes, List[AccountMeta]]:
        """
        Build SELL instruction data and accounts.

        Args:
            user_pubkey: User's wallet pubkey
            mint: Token mint address
            token_amount: Amount of tokens to sell
            min_sol_output: Minimum SOL to receive (slippage protection)

        Returns:
            (instruction_data, account_metas)
        """
        # Derive PDAs
        bonding_curve = derive_bonding_curve_pda(mint)
        associated_bonding_curve = derive_associated_bonding_curve(mint, bonding_curve)
        user_ata = derive_user_ata(user_pubkey, mint)

        # Instruction data: discriminator + token_amount (u64) + min_sol_output (u64)
        data = SELL_DISCRIMINATOR
        data += struct.pack("<Q", token_amount)  # u64 little-endian
        data += struct.pack("<Q", min_sol_output)  # u64 little-endian

        # Account metas
        accounts = [
            AccountMeta(GLOBAL_STATE, False, False),
            AccountMeta(FEE_RECIPIENT, False, True),
            AccountMeta(mint, False, False),
            AccountMeta(bonding_curve, False, True),
            AccountMeta(associated_bonding_curve, False, True),
            AccountMeta(user_ata, False, True),
            AccountMeta(user_pubkey, True, True),
            AccountMeta(SYSTEM_PROGRAM, False, False),
            AccountMeta(TOKEN_PROGRAM_ID, False, False),
            AccountMeta(EVENT_AUTHORITY, False, False),
            AccountMeta(PUMPFUN_PROGRAM_ID, False, False),
        ]

        return (data, accounts)


class SolanaTransactionBuilder:
    """
    Full Solana transaction builder with priority fees.

    Example:
        builder = SolanaTransactionBuilder(keypair)
        tx = builder.build_buy_tx(mint, token_amount, max_sol)
        sig = await builder.send_transaction(tx)
    """

    def __init__(
        self,
        keypair,  # solders.Keypair
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        priority_fee_lamports: int = 100_000,  # 0.0001 SOL
    ):
        self.keypair = keypair
        self.rpc_url = rpc_url
        self.priority_fee = priority_fee_lamports

    async def build_buy_transaction(
        self,
        mint: str,
        token_amount: int,
        max_sol_cost: int,
    ):
        """
        Build complete buy transaction with compute budget.

        Returns ready-to-sign transaction.
        """
        try:
            from solders.transaction import Transaction
            from solders.instruction import Instruction, AccountMeta as SoldersAccountMeta
            from solders.pubkey import Pubkey
            from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
            from solana.rpc.async_api import AsyncClient

            user_pubkey = str(self.keypair.pubkey())

            # Build pump.fun instruction
            data, accounts = PumpfunInstructionBuilder.build_buy_instruction(
                user_pubkey, mint, token_amount, max_sol_cost
            )

            # Convert to solders format
            solders_accounts = [
                SoldersAccountMeta(
                    Pubkey.from_string(acc.pubkey),
                    is_signer=acc.is_signer,
                    is_writable=acc.is_writable
                )
                for acc in accounts
            ]

            pumpfun_ix = Instruction(
                program_id=Pubkey.from_string(PUMPFUN_PROGRAM_ID),
                data=data,
                accounts=solders_accounts
            )

            # Add compute budget instructions for priority
            compute_limit_ix = set_compute_unit_limit(200_000)
            compute_price_ix = set_compute_unit_price(self.priority_fee)

            # Get recent blockhash
            async with AsyncClient(self.rpc_url) as client:
                blockhash_resp = await client.get_latest_blockhash()
                recent_blockhash = blockhash_resp.value.blockhash

            # Build transaction
            tx = Transaction.new_signed_with_payer(
                [compute_limit_ix, compute_price_ix, pumpfun_ix],
                payer=self.keypair.pubkey(),
                signing_keypairs=[self.keypair],
                recent_blockhash=recent_blockhash
            )

            return tx

        except ImportError as e:
            raise ImportError(f"Missing Solana SDK: pip install solana solders. Error: {e}")

    async def build_sell_transaction(
        self,
        mint: str,
        token_amount: int,
        min_sol_output: int,
    ):
        """
        Build complete sell transaction with compute budget.

        Returns ready-to-sign transaction.
        """
        try:
            from solders.transaction import Transaction
            from solders.instruction import Instruction, AccountMeta as SoldersAccountMeta
            from solders.pubkey import Pubkey
            from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
            from solana.rpc.async_api import AsyncClient

            user_pubkey = str(self.keypair.pubkey())

            # Build pump.fun instruction
            data, accounts = PumpfunInstructionBuilder.build_sell_instruction(
                user_pubkey, mint, token_amount, min_sol_output
            )

            # Convert to solders format
            solders_accounts = [
                SoldersAccountMeta(
                    Pubkey.from_string(acc.pubkey),
                    is_signer=acc.is_signer,
                    is_writable=acc.is_writable
                )
                for acc in accounts
            ]

            pumpfun_ix = Instruction(
                program_id=Pubkey.from_string(PUMPFUN_PROGRAM_ID),
                data=data,
                accounts=solders_accounts
            )

            # Add compute budget instructions
            compute_limit_ix = set_compute_unit_limit(150_000)
            compute_price_ix = set_compute_unit_price(self.priority_fee)

            # Get recent blockhash
            async with AsyncClient(self.rpc_url) as client:
                blockhash_resp = await client.get_latest_blockhash()
                recent_blockhash = blockhash_resp.value.blockhash

            # Build transaction
            tx = Transaction.new_signed_with_payer(
                [compute_limit_ix, compute_price_ix, pumpfun_ix],
                payer=self.keypair.pubkey(),
                signing_keypairs=[self.keypair],
                recent_blockhash=recent_blockhash
            )

            return tx

        except ImportError as e:
            raise ImportError(f"Missing Solana SDK: pip install solana solders. Error: {e}")

    async def send_transaction(self, tx) -> Optional[str]:
        """
        Send transaction to network.

        Returns:
            Transaction signature on success, None on failure
        """
        try:
            from solana.rpc.async_api import AsyncClient
            from solana.rpc.commitment import Confirmed

            async with AsyncClient(self.rpc_url) as client:
                result = await client.send_transaction(
                    tx,
                    opts={"skip_preflight": False, "preflight_commitment": Confirmed}
                )

                if result.value:
                    return str(result.value)
                return None

        except Exception as e:
            print(f"Transaction failed: {e}")
            return None


def decode_pumpfun_instruction(data: bytes) -> dict:
    """
    Decode pump.fun instruction data.

    Args:
        data: Raw instruction data bytes

    Returns:
        Decoded instruction details
    """
    if len(data) < 8:
        return {"type": "unknown", "error": "data too short"}

    discriminator = data[:8]

    if discriminator == BUY_DISCRIMINATOR:
        if len(data) >= 24:
            token_amount = struct.unpack("<Q", data[8:16])[0]
            max_sol = struct.unpack("<Q", data[16:24])[0]
            return {
                "type": "buy",
                "token_amount": token_amount,
                "max_sol_lamports": max_sol,
                "max_sol": max_sol / 1e9
            }
        return {"type": "buy", "error": "incomplete data"}

    elif discriminator == SELL_DISCRIMINATOR:
        if len(data) >= 24:
            token_amount = struct.unpack("<Q", data[8:16])[0]
            min_sol = struct.unpack("<Q", data[16:24])[0]
            return {
                "type": "sell",
                "token_amount": token_amount,
                "min_sol_lamports": min_sol,
                "min_sol": min_sol / 1e9
            }
        return {"type": "sell", "error": "incomplete data"}

    elif discriminator == CREATE_DISCRIMINATOR:
        return {"type": "create", "note": "token creation"}

    return {"type": "unknown", "discriminator": discriminator.hex()}


# Quick test
if __name__ == "__main__":
    # Test PDA derivation
    test_mint = "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"

    try:
        bonding_curve = derive_bonding_curve_pda(test_mint)
        print(f"Bonding curve PDA: {bonding_curve}")
    except Exception as e:
        print(f"PDA derivation error (expected without full ed25519): {e}")

    # Test instruction building
    user = "YourPubkeyHere123456789012345678901234567890"
    data, accounts = PumpfunInstructionBuilder.build_buy_instruction(
        user, test_mint, 1000000, 100000000  # 1M tokens, 0.1 SOL max
    )
    print(f"Buy instruction data: {data.hex()}")
    print(f"Account count: {len(accounts)}")

    # Test decode
    decoded = decode_pumpfun_instruction(data)
    print(f"Decoded: {decoded}")
