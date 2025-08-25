from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy import create_engine

from .settings import settings


class Base(DeclarativeBase):
    pass


engine: Engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def create_extensions_and_indexes() -> None:
    with engine.begin() as conn:
        # Create extensions if permitted; ignore if lacking privileges
        try:
            conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception:
            pass
        try:
            conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        except Exception:
            pass
        # Create indexes if tables exist; if not, it will be re-tried after first create_all
        try:
            conn.exec_driver_sql(
                """
                DO $$
                BEGIN
                    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='chunks') THEN
                        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat
                        ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                        CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
                        CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks (created_at DESC);
                    END IF;
                END$$;
                """
            )
        except Exception:
            pass


def init_db() -> None:
    # Import models to register metadata
    from . import models  # noqa: F401

    # Ensure required extensions exist before creating tables that depend on them
    create_extensions_and_indexes()
    Base.metadata.create_all(bind=engine)
    run_migrations()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def db_healthcheck() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def run_migrations() -> None:
    """Lightweight in-app migration runner to manage schema changes safely without Alembic files.

    Stores applied migration IDs in table `schema_migrations` and executes idempotent SQL blocks.
    """
    with engine.begin() as conn:
        # Ensure migrations table exists
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )

        # Fetch applied ids
        applied = {row[0] for row in conn.exec_driver_sql("SELECT id FROM schema_migrations").fetchall()}

        migrations: list[tuple[str, str]] = []

        # Drop legacy trigram index if present (we now rely on ORM-defined GIN index)
        migrations.append((
            "2024-08-01-01-drop-old-trgm",
            "DROP INDEX IF EXISTS idx_chunks_text_trgm;"
        ))

        # Create tsvector index to support tsvector lexical method
        migrations.append((
            "2024-08-01-02-create-tsvector-index",
            "CREATE INDEX IF NOT EXISTS idx_chunks_text_tsv ON chunks USING gin (to_tsvector('english', text));"
        ))

        # Ensure ivfflat index exists with a reasonable lists parameter
        migrations.append((
            "2024-08-01-03-ensure-ivfflat",
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
        ))

        for mig_id, sql in migrations:
            if mig_id in applied:
                continue
            try:
                conn.exec_driver_sql(sql)
                conn.exec_driver_sql("INSERT INTO schema_migrations (id) VALUES (:id)", {"id": mig_id})
            except Exception:
                # Do not block startup; leave unapplied for next run
                pass
