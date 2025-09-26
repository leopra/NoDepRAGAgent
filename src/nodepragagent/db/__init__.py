"""Database utilities for working with the transactional store."""

from __future__ import annotations

import os
from typing import List

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.dialects import postgresql

from .models import Base

DEFAULT_POSTGRES_URL = "postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db"


def postgres_url() -> str:
    """Return the configured Postgres connection string."""

    return os.getenv("POSTGRES_URL", DEFAULT_POSTGRES_URL)


def create_postgres_engine() -> Engine:
    """Instantiate a SQLAlchemy engine for Postgres."""

    return create_engine(postgres_url(), future=True)


def schema_summary() -> str:
    """Produce a compact textual overview of the database schema."""

    dialect = postgresql.dialect()
    lines: List[str] = []
    for table in Base.metadata.sorted_tables:
        lines.append(f"Table {table.name}")
        for column in table.columns:
            type_repr = column.type.compile(dialect=dialect)
            nullability = "NOT NULL" if not column.nullable else "NULLABLE"
            default = column.default
            default_repr = f" DEFAULT {default.arg}" if default is not None else ""
            pk_flag = " PRIMARY KEY" if column.primary_key else ""
            fk = next(iter(column.foreign_keys), None)
            fk_repr = f" REFERENCES {fk.target_fullname}" if fk is not None else ""
            lines.append(f"  - {column.name} {type_repr}{pk_flag} {nullability}{default_repr}{fk_repr}".rstrip())
        lines.append("")
    return "\n".join(lines).strip()
