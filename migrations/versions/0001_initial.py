"""Initial migration — creates users, cycle_logs, chat_messages tables.

Revision ID: 0001_initial
Revises:
Create Date: 2025-01-01
"""

from alembic import op
import sqlalchemy as sa

revision      = "0001_initial"
down_revision = None
branch_labels = None
depends_on    = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id",           sa.Integer(),               nullable=False),
        sa.Column("name",         sa.String(120),             nullable=False),
        sa.Column("email",        sa.String(200),             nullable=False),
        sa.Column("password",     sa.String(300),             nullable=False),
        sa.Column("age",          sa.Integer(),               server_default="25"),
        sa.Column("avg_cycle",    sa.Float(),                 server_default="28.0"),
        sa.Column("bmi",          sa.Float(),                 server_default="22.5"),
        sa.Column("is_irregular", sa.Boolean(),               server_default="0"),
        sa.Column("created_at",   sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_id",    "users", ["id"],    unique=False)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "cycle_logs",
        sa.Column("id",              sa.Integer(),               nullable=False),
        sa.Column("user_id",         sa.Integer(),               nullable=False),
        sa.Column("log_date",        sa.Date(),                  nullable=False),
        sa.Column("flow_intensity",  sa.String(20),              server_default="none"),
        sa.Column("mood",            sa.String(30),              server_default="neutral"),
        sa.Column("symptoms",        sa.Text(),                  nullable=True),
        sa.Column("stress",          sa.String(20),              server_default="medium"),
        sa.Column("sleep",           sa.String(20),              server_default="normal"),
        sa.Column("exercise",        sa.String(20),              server_default="okay"),
        sa.Column("notes",           sa.Text(),                  server_default=""),
        sa.Column("cycle_day",       sa.Integer(),               server_default="1"),
        sa.Column("days_since_last", sa.Integer(),               nullable=True),
        sa.Column("created_at",      sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_cycle_logs_id", "cycle_logs", ["id"], unique=False)

    op.create_table(
        "chat_messages",
        sa.Column("id",         sa.Integer(),               nullable=False),
        sa.Column("user_id",    sa.Integer(),               nullable=False),
        sa.Column("role",       sa.String(20),              nullable=True),
        sa.Column("content",    sa.Text(),                  nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_messages_id", "chat_messages", ["id"], unique=False)


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("cycle_logs")
    op.drop_table("users")
