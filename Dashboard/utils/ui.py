"""Shared UI helpers for the dashboard (hero, styling, sidebar)."""
from __future__ import annotations
from pathlib import Path
import streamlit as st

_CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"


def apply_style() -> None:
    """Inject the custom stylesheet. Safe to call on every page."""
    if _CSS_PATH.exists():
        st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, icon: str = "") -> None:
    """Gradient banner rendered at the top of each page."""
    icon_html = f'<span class="hero-icon">{icon}</span>' if icon else ""
    st.markdown(
        f"""
        <div class="hero">
            <h1>{icon_html}{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, icon: str = "") -> None:
    """Underlined section header used between chart blocks."""
    icon_html = f"{icon} " if icon else ""
    st.markdown(
        f'<div class="section-header"><h3>{icon_html}{title}</h3></div>',
        unsafe_allow_html=True,
    )


def sidebar_brand(total_reviews: int | None = None,
                  fake_count: int | None = None,
                  best_acc: float | None = None) -> None:
    """Branded sidebar with a summary stat block."""
    st.sidebar.markdown(
        '<div class="sidebar-brand">Amazon Reviews Analysis</div>'
        '<div class="sidebar-sub">EDA · Fake Detection · Sentiment</div>',
        unsafe_allow_html=True,
    )
    if total_reviews is not None:
        st.sidebar.markdown(
            f'<div class="sidebar-stat">'
            f'<div class="label">Total Reviews</div>'
            f'<div class="value">{total_reviews:,}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if fake_count is not None:
        st.sidebar.markdown(
            f'<div class="sidebar-stat">'
            f'<div class="label">Fake (Moderate)</div>'
            f'<div class="value">{fake_count:,}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if best_acc is not None:
        st.sidebar.markdown(
            f'<div class="sidebar-stat">'
            f'<div class="label">Best Model Acc.</div>'
            f'<div class="value">{best_acc:.2%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def pipeline_chips(steps: list[str]) -> None:
    """Render a pipeline as styled chips with arrows."""
    parts = []
    for i, step in enumerate(steps):
        parts.append(f'<span class="pipe-chip">{step}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="pipe-arrow">&rarr;</span>')
    st.markdown(
        f'<div class="pipeline-row">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )
