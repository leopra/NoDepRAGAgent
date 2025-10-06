"""Database utilities for working with the transactional store."""

from __future__ import annotations

import os
import json
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .models import Base

DEFAULT_POSTGRES_URL = "postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db"


def postgres_url() -> str:
    """Return the configured Postgres connection string."""

    return os.getenv("POSTGRES_URL", DEFAULT_POSTGRES_URL)


def create_postgres_engine() -> Engine:
    """Instantiate a SQLAlchemy engine for Postgres."""

    return create_engine(postgres_url(), future=True)

