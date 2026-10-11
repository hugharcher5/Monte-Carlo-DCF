"""Small helpers shared by the page sections."""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st


def pct(v: float, decimals: int = 1, sign: bool = False) -> str:
    if v is None or not math.isfinite(v):
        return "n/a"
    return f"{v:+.{decimals}%}" if sign else f"{v:.{decimals}%}"


def usd(v: float, decimals: int = 2) -> str:
    """Dollar amount escaped for Streamlit markdown (a bare $ pair renders as LaTeX)."""
    if v is None or not math.isfinite(v):
        return "n/a"
    return f"\\${v:,.{decimals}f}"


def table(rows, columns, width: str = "stretch") -> None:
    st.dataframe(pd.DataFrame(rows, columns=columns), width=width, hide_index=True)
