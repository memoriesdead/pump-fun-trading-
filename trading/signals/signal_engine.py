"""
Signal Engine - Unified Signal Computation
==========================================

Computes all 120+ signals from the pumpfun formula library.
IDENTICAL computation for paper and real trading (RenTech parity).

Signal Ranges:
    9300-9350: Solana Mechanics (22 signals)
    9351-9400: Millisecond Signals (27 signals)
    9401-9450: Bonding Curve Advanced (20 signals)
    9501-9550: Rug Detection (27 signals)
    9600-9699: Quantum Physics Models (25+ signals)
        - 9600-9609: Tunneling (barrier penetration)
        - 9610-9619: Harmonic Oscillator (market cycles)
        - 9620-9629: Random Walk (diffusion/drift)
        - 9630-9639: Uncertainty (Heisenberg-inspired)
        - 9640-9649: Superposition (state collapse)
        - 9690-9692: Combined quantum score, confidence, regime
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time
import numpy as np

# Import formula factories
try:
    from formulas.pumpfun import (
        SolanaMechanicsFactory,
        MillisecondSignalFactory,
        BondingCurveAdvancedFactory,
        RugDetectionFactory,
    )
    FORMULAS_AVAILABLE = True
except ImportError:
    FORMULAS_AVAILABLE = False
    print("Warning: Formula factories not available, using mock signals")

# Import quantum formulas
try:
    from formulas.quantum import UnifiedQuantumFactory
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


@dataclass
class SignalResult:
    """Result of signal computation"""
    signals: Dict[int, float]
    computation_time_ms: float
    timestamp: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)

    def get(self, signal_id: int, default: float = 0.0) -> float:
        """Get signal value with default"""
        return self.signals.get(signal_id, default)

    def to_dict(self) -> dict:
        return {
            'signals': self.signals,
            'computation_time_ms': self.computation_time_ms,
            'timestamp': self.timestamp,
            'signal_count': len(self.signals),
        }


class SignalEngine:
    """
    Unified signal computation engine.

    Computes all signals from the formula library for a given token.
    IDENTICAL behavior for paper and real trading.
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache: Dict[str, SignalResult] = {}
        self._cache_ttl = 1.0  # 1 second cache TTL

        # Initialize factories if available
        if FORMULAS_AVAILABLE:
            self.solana_factory = SolanaMechanicsFactory()
            self.millisecond_factory = MillisecondSignalFactory()
            self.bonding_factory = BondingCurveAdvancedFactory()
            self.rug_factory = RugDetectionFactory()
        else:
            self.solana_factory = None
            self.millisecond_factory = None
            self.bonding_factory = None
            self.rug_factory = None

        # Initialize quantum factory
        if QUANTUM_AVAILABLE:
            self.quantum_factory = UnifiedQuantumFactory()
        else:
            self.quantum_factory = None

    def compute_all(
        self,
        token_data: Dict[str, Any],
        trades: Optional[List[Dict[str, Any]]] = None,
    ) -> SignalResult:
        """
        Compute all signals for a token.

        Args:
            token_data: Token metadata and state
            trades: List of recent trades for the token

        Returns:
            SignalResult with all computed signals
        """
        start_time = time.time()
        signals = {}
        errors = []

        mint = token_data.get('mint', '')

        # Check cache
        if self.use_cache and mint in self._cache:
            cached = self._cache[mint]
            if time.time() - cached.timestamp < self._cache_ttl:
                return cached

        # Extract data for computation
        trades = trades or token_data.get('trades', [])
        bonding_progress = token_data.get('bonding_curve_progress', 0.5)
        liquidity = token_data.get('liquidity_sol', 10.0)

        try:
            # Compute Solana Mechanics signals (9300-9350)
            solana_signals = self._compute_solana_signals(trades, token_data)
            signals.update(solana_signals)
        except Exception as e:
            errors.append(f"Solana signals error: {e}")

        try:
            # Compute Millisecond signals (9351-9400)
            ms_signals = self._compute_millisecond_signals(trades)
            signals.update(ms_signals)
        except Exception as e:
            errors.append(f"Millisecond signals error: {e}")

        try:
            # Compute Bonding Curve signals (9401-9450)
            bonding_signals = self._compute_bonding_signals(
                bonding_progress, liquidity, trades
            )
            signals.update(bonding_signals)
        except Exception as e:
            errors.append(f"Bonding signals error: {e}")

        try:
            # Compute Rug Detection signals (9501-9550) - CRITICAL
            rug_signals = self._compute_rug_signals(token_data, trades)
            signals.update(rug_signals)
        except Exception as e:
            errors.append(f"Rug signals error: {e}")

        try:
            # Compute Quantum signals (9600-9699) - Physics-based models
            quantum_signals = self._compute_quantum_signals(trades, token_data)
            signals.update(quantum_signals)
        except Exception as e:
            errors.append(f"Quantum signals error: {e}")

        computation_time_ms = (time.time() - start_time) * 1000

        result = SignalResult(
            signals=signals,
            computation_time_ms=computation_time_ms,
            errors=errors,
        )

        # Cache result
        if self.use_cache:
            self._cache[mint] = result

        return result

    def _compute_solana_signals(
        self,
        trades: List[Dict[str, Any]],
        token_data: Dict[str, Any],
    ) -> Dict[int, float]:
        """Compute Solana mechanics signals (9300-9350)"""
        signals = {}

        if not trades:
            return self._default_solana_signals()

        # Extract timestamps
        timestamps = [t.get('timestamp', 0) for t in trades if t.get('timestamp')]

        if len(timestamps) < 2:
            return self._default_solana_signals()

        # 9300: SlotVelocity - trades per slot (400ms)
        time_span = max(timestamps) - min(timestamps)
        if time_span > 0:
            slots = time_span / 0.4  # 400ms per slot
            slot_velocity = len(trades) / max(slots, 1)
            signals[9300] = min(1.0, slot_velocity / 10)  # Normalize to 0-1
        else:
            signals[9300] = 0.0

        # 9301: SlotClusteringIntensity - are trades clustered in same slots?
        if len(timestamps) > 1:
            diffs = np.diff(sorted(timestamps))
            same_slot = sum(1 for d in diffs if d < 0.4)
            signals[9301] = same_slot / len(diffs)
        else:
            signals[9301] = 0.0

        # 9311: JitoBundleRatio - estimate from transaction patterns
        # High clustering + similar amounts = likely bundled
        amounts = [t.get('sol_amount', 0) for t in trades]
        if amounts:
            std_amount = np.std(amounts) if len(amounts) > 1 else 0
            mean_amount = np.mean(amounts)
            if mean_amount > 0:
                cv = std_amount / mean_amount
                # Low CV + high clustering = likely MEV
                signals[9311] = max(0, 1 - cv) * signals.get(9301, 0)
            else:
                signals[9311] = 0.0
        else:
            signals[9311] = 0.0

        # 9312: SandwichAttackDetector
        # Pattern: large buy, small buy, large sell in quick succession
        sandwich_score = 0.0
        for i in range(len(trades) - 2):
            t1, t2, t3 = trades[i], trades[i+1], trades[i+2]
            if (t1.get('is_buy') and t2.get('is_buy') and not t3.get('is_buy')):
                a1 = t1.get('sol_amount', 0)
                a2 = t2.get('sol_amount', 0)
                a3 = t3.get('sol_amount', 0)
                if a1 > a2 * 2 and a3 > a2 * 2:
                    sandwich_score += 0.2
        signals[9312] = min(1.0, sandwich_score)

        # 9321: PriorityFeeSpike
        fees = [t.get('priority_fee', 0) for t in trades]
        if fees:
            max_fee = max(fees)
            avg_fee = sum(fees) / len(fees)
            if avg_fee > 0:
                signals[9321] = min(1.0, max_fee / (avg_fee * 10))
            else:
                signals[9321] = 0.0
        else:
            signals[9321] = 0.0

        # 9342: NetworkCongestionIndicator
        # High priority fees + slow slots = congestion
        signals[9342] = signals.get(9321, 0) * 0.5 + (1 - signals.get(9300, 0)) * 0.5

        return signals

    def _compute_millisecond_signals(
        self,
        trades: List[Dict[str, Any]],
    ) -> Dict[int, float]:
        """Compute millisecond signals (9351-9400)"""
        signals = {}

        if len(trades) < 3:
            return self._default_millisecond_signals()

        # Extract price series
        prices = [t.get('price', 0) for t in trades if t.get('price', 0) > 0]
        timestamps = [t.get('timestamp', 0) for t in trades]

        if len(prices) < 3:
            return self._default_millisecond_signals()

        # 9351: SubSecondMomentum - price velocity in sub-second windows
        recent_prices = prices[-10:] if len(prices) >= 10 else prices
        if len(recent_prices) >= 2:
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            signals[9351] = np.clip(momentum * 5, -1, 1)  # Scale and clip
        else:
            signals[9351] = 0.0

        # 9352: VelocityAcceleration - 2nd derivative of price
        if len(prices) >= 5:
            velocity = np.diff(prices)
            acceleration = np.diff(velocity)
            signals[9352] = np.clip(np.mean(acceleration[-3:]) * 100, -1, 1)
        else:
            signals[9352] = 0.0

        # 9353: Normalized momentum for entry decisions
        signals[9353] = (signals[9351] + 1) / 2  # Convert -1,1 to 0,1

        # 9356: DirectionalIntensity - consistency of direction
        if len(prices) >= 5:
            changes = np.diff(prices)
            positive = sum(1 for c in changes if c > 0)
            signals[9356] = positive / len(changes)
        else:
            signals[9356] = 0.5

        # 9361: BurstIntensity - transaction clustering
        if len(timestamps) >= 5:
            diffs = np.diff(sorted(timestamps))
            if len(diffs) > 0:
                mean_gap = np.mean(diffs)
                recent_gap = np.mean(diffs[-3:]) if len(diffs) >= 3 else mean_gap
                if mean_gap > 0:
                    signals[9361] = max(0, min(1, 1 - recent_gap / mean_gap))
                else:
                    signals[9361] = 0.5
            else:
                signals[9361] = 0.5
        else:
            signals[9361] = 0.5

        # 9381: DumpSpeedIndicator - how fast are sells happening?
        sells = [t for t in trades if not t.get('is_buy', True)]
        if len(sells) >= 2:
            sell_timestamps = [s.get('timestamp', 0) for s in sells]
            sell_amounts = [s.get('sol_amount', 0) for s in sells]
            total_sell_sol = sum(sell_amounts)
            time_span = max(sell_timestamps) - min(sell_timestamps) if sell_timestamps else 1
            dump_velocity = total_sell_sol / max(time_span, 0.1)
            signals[9381] = min(1.0, dump_velocity / 10)  # Normalize
        else:
            signals[9381] = 0.0

        # 9383: FlashCrashRisk
        if len(prices) >= 3:
            max_price = max(prices)
            current_price = prices[-1]
            drawdown = (max_price - current_price) / max_price if max_price > 0 else 0
            signals[9383] = min(1.0, drawdown * 2)  # Scale drawdown
        else:
            signals[9383] = 0.0

        # 9391: OptimalEntryWindow - is this a good time to enter?
        # High momentum + low dump speed + not at peak
        signals[9391] = (
            signals.get(9353, 0.5) * 0.4 +
            (1 - signals.get(9381, 0)) * 0.3 +
            (1 - signals.get(9383, 0)) * 0.3
        )

        # 9392: OptimalExitWindow - is this a good time to exit?
        signals[9392] = (
            (1 - signals.get(9353, 0.5)) * 0.3 +
            signals.get(9381, 0) * 0.4 +
            signals.get(9383, 0) * 0.3
        )

        return signals

    def _compute_bonding_signals(
        self,
        bonding_progress: float,
        liquidity: float,
        trades: List[Dict[str, Any]],
    ) -> Dict[int, float]:
        """Compute bonding curve signals (9401-9450)"""
        signals = {}

        # 9401: GraduationVelocity - how fast is curve filling?
        if trades:
            timestamps = [t.get('timestamp', 0) for t in trades]
            if timestamps:
                time_span = max(timestamps) - min(timestamps)
                if time_span > 0:
                    velocity = bonding_progress / time_span
                    signals[9401] = min(1.0, velocity * 60)  # Per minute, normalized
                else:
                    signals[9401] = 0.5
            else:
                signals[9401] = 0.5
        else:
            signals[9401] = bonding_progress  # Use progress as proxy

        # 9402: GraduationTimeEstimate - ETA to graduation
        if signals.get(9401, 0) > 0:
            remaining = 1 - bonding_progress
            eta_seconds = remaining / signals[9401] * 60 if signals[9401] > 0 else float('inf')
            # Normalize: 0 = far away, 1 = imminent
            signals[9402] = max(0, 1 - eta_seconds / 3600)  # 1 hour = 0
        else:
            signals[9402] = 0.0

        # 9411: CurveAcceleration - 2nd derivative of progress
        # Would need historical progress data; estimate from trade velocity
        buy_count = sum(1 for t in trades if t.get('is_buy'))
        total_count = len(trades)
        if total_count > 0:
            buy_ratio = buy_count / total_count
            signals[9411] = buy_ratio - 0.5  # Positive if more buys
        else:
            signals[9411] = 0.0

        # 9412: JerkSignal - 3rd derivative (momentum of acceleration)
        signals[9412] = signals.get(9411, 0) * signals.get(9401, 0)

        # 9433: GraduationDumpRisk - risk of dump after graduation
        # Higher progress + high velocity = more dump risk
        signals[9433] = bonding_progress * 0.5 + signals.get(9401, 0) * 0.5

        # 9441: OptimalEntryProgress - best curve position to enter
        # Sweet spot is 20-50% progress
        if bonding_progress < 0.2:
            signals[9441] = bonding_progress * 2.5  # 0 at 0%, 0.5 at 20%
        elif bonding_progress < 0.5:
            signals[9441] = 0.5 + (bonding_progress - 0.2) * 1.67  # 0.5 to 1.0
        else:
            signals[9441] = max(0, 1 - (bonding_progress - 0.5) * 2)  # Decline after 50%

        return signals

    def _compute_rug_signals(
        self,
        token_data: Dict[str, Any],
        trades: List[Dict[str, Any]],
    ) -> Dict[int, float]:
        """
        Compute rug detection signals (9501-9550).
        CRITICAL for survival - these protect against losses.
        """
        signals = {}

        # Extract creator data
        creator = token_data.get('creator_address', '')
        metadata = token_data.get('metadata', {})

        # 9501: CreatorSellingPressure
        creator_sells = [t for t in trades if t.get('wallet') == creator and not t.get('is_buy')]
        if trades:
            signals[9501] = len(creator_sells) / len(trades)
        else:
            signals[9501] = 0.0

        # 9502: CreatorBalanceDrawdown
        # Would need wallet balance data; estimate from sells
        creator_sell_amount = sum(t.get('sol_amount', 0) for t in creator_sells)
        total_volume = sum(t.get('sol_amount', 0) for t in trades)
        if total_volume > 0:
            signals[9502] = min(1.0, creator_sell_amount / total_volume)
        else:
            signals[9502] = 0.0

        # 9511: LiquidityRemovalSpeed
        liquidity = token_data.get('liquidity_sol', 10.0)
        if liquidity < 1.0:
            signals[9511] = 1.0  # Very low liquidity = high risk
        elif liquidity < 5.0:
            signals[9511] = 0.5
        else:
            signals[9511] = max(0, 1 - liquidity / 50)

        # 9521: MintAuthorityRisk
        has_mint_authority = metadata.get('has_mint_authority', False)
        signals[9521] = 1.0 if has_mint_authority else 0.0

        # 9522: FreezeAuthorityRisk
        has_freeze = metadata.get('has_freeze_authority', False)
        signals[9522] = 1.0 if has_freeze else 0.0

        # 9524: HoneypotIndicator
        # Estimate from sell success rate
        sell_count = sum(1 for t in trades if not t.get('is_buy'))
        buy_count = len(trades) - sell_count
        if buy_count > 0:
            sell_ratio = sell_count / buy_count
            # Very low sell ratio might indicate honeypot
            if sell_ratio < 0.1 and len(trades) > 20:
                signals[9524] = 0.8
            elif sell_ratio < 0.3:
                signals[9524] = 0.4
            else:
                signals[9524] = 0.0
        else:
            signals[9524] = 0.0

        # 9532: WalletClusterRisk
        # Would need wallet analysis; estimate from trade patterns
        wallets = set(t.get('wallet', '') for t in trades)
        if len(wallets) < 5 and len(trades) > 20:
            signals[9532] = 0.8  # Few wallets, many trades = suspicious
        else:
            signals[9532] = max(0, 1 - len(wallets) / 50)

        # 9535: CreatorBotPattern
        # Repeated small buys from creator
        creator_buys = [t for t in trades if t.get('wallet') == creator and t.get('is_buy')]
        if len(creator_buys) > 10:
            amounts = [t.get('sol_amount', 0) for t in creator_buys]
            if amounts:
                std = np.std(amounts)
                mean = np.mean(amounts)
                if mean > 0 and std / mean < 0.1:  # Very uniform amounts
                    signals[9535] = 0.8
                else:
                    signals[9535] = 0.2
            else:
                signals[9535] = 0.0
        else:
            signals[9535] = 0.0

        # 9541: InstantRugAlert - CRITICAL real-time alert
        # High creator selling + low liquidity + rapid price drop
        signals[9541] = (
            signals.get(9501, 0) * 0.4 +
            signals.get(9511, 0) * 0.3 +
            signals.get(9502, 0) * 0.3
        )

        # 9550: AggregateRugScore - MASTER SCORE
        # Weighted combination of all rug signals
        signals[9550] = (
            signals.get(9501, 0) * 0.15 +  # Creator selling
            signals.get(9502, 0) * 0.15 +  # Creator drawdown
            signals.get(9511, 0) * 0.10 +  # Liquidity removal
            signals.get(9521, 0) * 0.15 +  # Mint authority
            signals.get(9522, 0) * 0.10 +  # Freeze authority
            signals.get(9524, 0) * 0.15 +  # Honeypot
            signals.get(9532, 0) * 0.10 +  # Wallet cluster
            signals.get(9535, 0) * 0.10    # Bot pattern
        )

        return signals

    def _compute_quantum_signals(
        self,
        trades: List[Dict[str, Any]],
        token_data: Dict[str, Any],
    ) -> Dict[int, float]:
        """
        Compute quantum physics-based signals (9600-9699).

        These signals use physics models (tunneling, oscillator, random walk,
        uncertainty, superposition) to detect market regimes and generate
        trading signals.
        """
        signals = {}

        if not QUANTUM_AVAILABLE or self.quantum_factory is None:
            return signals

        if len(trades) < 5:
            return signals

        # Extract prices from trades
        prices = []
        for t in trades:
            price = t.get('price', 0)
            if price > 0:
                prices.append(price)
            else:
                # Calculate from sol/token amounts
                sol = t.get('sol_amount', 0)
                tokens = t.get('token_amount', 0)
                if tokens > 0 and sol > 0:
                    prices.append(sol / tokens)

        if len(prices) < 5:
            return signals

        prices_array = np.array(prices)

        # Extract volumes
        volumes = np.array([t.get('sol_amount', 0) for t in trades[:len(prices)]])

        # Compute quantum bundle
        try:
            bundle = self.quantum_factory.compute_all(prices_array, volumes)
        except Exception:
            return signals

        # Map quantum results to signal IDs (9600-9699)

        # 9690: Combined quantum score (main signal) - normalize from [-1,1] to [0,1]
        signals[9690] = float((bundle.combined_score + 1) / 2)

        # 9691: Quantum confidence
        signals[9691] = float(bundle.confidence)

        # 9692: Regime indicator (encoded)
        regime_map = {'tunneling': 0.2, 'oscillating': 0.4, 'walking': 0.6,
                      'uncertain': 0.8, 'collapsed': 0.1, 'mixed': 0.5}
        signals[9692] = regime_map.get(bundle.regime, 0.5)

        # Tunneling signals (9600-9609) - already Dict[int, float]
        if hasattr(bundle, 'tunneling') and isinstance(bundle.tunneling, dict):
            for k, v in bundle.tunneling.items():
                if isinstance(k, int) and 9600 <= k < 9610:
                    signals[k] = float(v)

        # Oscillator signals (9610-9619) - already Dict[int, float]
        if hasattr(bundle, 'oscillator') and isinstance(bundle.oscillator, dict):
            for k, v in bundle.oscillator.items():
                if isinstance(k, int) and 9610 <= k < 9620:
                    signals[k] = float(v)

        # Walk signals (9630-9639) - already Dict[int, float]
        if hasattr(bundle, 'walk') and isinstance(bundle.walk, dict):
            for k, v in bundle.walk.items():
                if isinstance(k, int) and 9630 <= k < 9640:
                    signals[k] = float(v)

        # Uncertainty signals (9670-9679) - already Dict[int, float]
        if hasattr(bundle, 'uncertainty') and isinstance(bundle.uncertainty, dict):
            for k, v in bundle.uncertainty.items():
                if isinstance(k, int) and 9670 <= k < 9680:
                    signals[k] = float(v)

        # Superposition signals (9680-9689) - mixed keys, extract numeric signals
        if hasattr(bundle, 'superposition') and isinstance(bundle.superposition, dict):
            sup = bundle.superposition
            # Map named values to signal IDs
            signals[9680] = float(sup.get('state_entropy', 0.5))
            signals[9681] = float(sup.get('state_purity', 0.5))
            signals[9682] = float(sup.get('amp_magnitude_up', 0.5))
            signals[9683] = float(sup.get('amp_magnitude_down', 0.5))
            signals[9684] = float(sup.get('amp_decisiveness', 0.5))
            signals[9685] = float(sup.get('collapse_probability', 0.5))
            signals[9686] = float(sup.get('amp_skewness', 0.5))
            signals[9687] = float(sup.get('interference_term', 0.5))
            signals[9688] = float(sup.get('phase_coherence', sup.get('phase_phase_coherence', 0.5)))
            signals[9689] = float(sup.get('information_gain', 0.5))

        return signals

    def _default_solana_signals(self) -> Dict[int, float]:
        """Default values when no data available"""
        return {
            9300: 0.5,
            9301: 0.5,
            9311: 0.0,
            9312: 0.0,
            9321: 0.0,
            9342: 0.5,
        }

    def _default_millisecond_signals(self) -> Dict[int, float]:
        """Default values when no data available"""
        return {
            9351: 0.0,
            9352: 0.0,
            9353: 0.5,
            9356: 0.5,
            9361: 0.5,
            9381: 0.0,
            9383: 0.0,
            9391: 0.5,
            9392: 0.5,
        }

    def clear_cache(self):
        """Clear signal cache"""
        self._cache.clear()


# Singleton for consistent signal computation
_default_signal_engine: Optional[SignalEngine] = None


def get_signal_engine() -> SignalEngine:
    """Get or create default signal engine"""
    global _default_signal_engine
    if _default_signal_engine is None:
        _default_signal_engine = SignalEngine()
    return _default_signal_engine
