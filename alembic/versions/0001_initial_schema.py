"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-03-26
"""

from alembic import op
import sqlalchemy as sa


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("genre", sa.String(length=120), nullable=True),
        sa.Column("tags", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_items_id"), "items", ["id"], unique=False)
    op.create_index(op.f("ix_items_title"), "items", ["title"], unique=False)

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column("variant", sa.String(length=80), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_model_versions_id"), "model_versions", ["id"], unique=False)
    op.create_index(op.f("ix_model_versions_is_active"), "model_versions", ["is_active"], unique=False)

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_id"), "users", ["id"], unique=False)

    op.create_table(
        "ab_assignments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("experiment_name", sa.String(length=120), nullable=False),
        sa.Column("variant", sa.String(length=40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "experiment_name", name="uq_user_experiment"),
    )
    op.create_index(op.f("ix_ab_assignments_experiment_name"), "ab_assignments", ["experiment_name"], unique=False)
    op.create_index(op.f("ix_ab_assignments_id"), "ab_assignments", ["id"], unique=False)
    op.create_index(op.f("ix_ab_assignments_user_id"), "ab_assignments", ["user_id"], unique=False)

    op.create_table(
        "ratings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("rating", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["item_id"], ["items.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "item_id", name="uq_user_item_rating"),
    )
    op.create_index(op.f("ix_ratings_id"), "ratings", ["id"], unique=False)
    op.create_index(op.f("ix_ratings_item_id"), "ratings", ["item_id"], unique=False)
    op.create_index(op.f("ix_ratings_user_id"), "ratings", ["user_id"], unique=False)

    op.create_table(
        "recommendation_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=True),
        sa.Column("experiment_variant", sa.String(length=40), nullable=False),
        sa.Column("event_type", sa.String(length=30), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["item_id"], ["items.id"]),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_recommendation_events_id"), "recommendation_events", ["id"], unique=False)
    op.create_index(op.f("ix_recommendation_events_item_id"), "recommendation_events", ["item_id"], unique=False)
    op.create_index(op.f("ix_recommendation_events_model_version_id"), "recommendation_events", ["model_version_id"], unique=False)
    op.create_index(op.f("ix_recommendation_events_user_id"), "recommendation_events", ["user_id"], unique=False)

    op.create_table(
        "user_features",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("click_count", sa.Integer(), nullable=False),
        sa.Column("impression_count", sa.Integer(), nullable=False),
        sa.Column("watch_seconds", sa.Float(), nullable=False),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_index(op.f("ix_user_features_user_id"), "user_features", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_user_features_user_id"), table_name="user_features")
    op.drop_table("user_features")
    op.drop_index(op.f("ix_recommendation_events_user_id"), table_name="recommendation_events")
    op.drop_index(op.f("ix_recommendation_events_model_version_id"), table_name="recommendation_events")
    op.drop_index(op.f("ix_recommendation_events_item_id"), table_name="recommendation_events")
    op.drop_index(op.f("ix_recommendation_events_id"), table_name="recommendation_events")
    op.drop_table("recommendation_events")
    op.drop_index(op.f("ix_ratings_user_id"), table_name="ratings")
    op.drop_index(op.f("ix_ratings_item_id"), table_name="ratings")
    op.drop_index(op.f("ix_ratings_id"), table_name="ratings")
    op.drop_table("ratings")
    op.drop_index(op.f("ix_ab_assignments_user_id"), table_name="ab_assignments")
    op.drop_index(op.f("ix_ab_assignments_id"), table_name="ab_assignments")
    op.drop_index(op.f("ix_ab_assignments_experiment_name"), table_name="ab_assignments")
    op.drop_table("ab_assignments")
    op.drop_index(op.f("ix_users_id"), table_name="users")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
    op.drop_index(op.f("ix_model_versions_is_active"), table_name="model_versions")
    op.drop_index(op.f("ix_model_versions_id"), table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_index(op.f("ix_items_title"), table_name="items")
    op.drop_index(op.f("ix_items_id"), table_name="items")
    op.drop_table("items")
