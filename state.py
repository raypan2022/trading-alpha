from typing import Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class TradingVerdict(BaseModel):
    ticker: str = Field(description="The financial asset ticker symbol analyzed.")
    action: str = Field(description="Definitive strategy decision. MUST be exactly: BUY, HOLD, or SELL.")
    target_price_30d: float = Field(description="Target price for a 30-day window.")
    confidence_score: float = Field(description="Confidence score bounded 0.0 to 1.0.")
    bull_concession: str = Field(description="The strongest bull point the Judge acknowledges as a real risk.")
    bear_concession: str = Field(description="The strongest bear point the Judge overrides with contrary evidence.")
    core_rationale: str = Field(description="Consolidated empirical justification for the final trade action.")


class AgentState(TypedDict):
    ticker: str
    as_of: Optional[str]          # ISO date for backtesting; None = live data
    market_regime: str            # shared macro context, set by the macro node
    bull_report: str
    bear_report: str
    final_verdict: Optional[TradingVerdict]
