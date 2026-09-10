"""Add learner quiz sessions table for M4A."""
from alembic import op
import sqlalchemy as sa

revision = "0004_learner_quiz_sessions"
down_revision = "0003_assessment_quality_evaluations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "learner_quiz_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("learner_id", sa.String(255), nullable=False),
        sa.Column("question_ids", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_learner_quiz_sessions_learner_id",
        "learner_quiz_sessions",
        ["learner_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_learner_quiz_sessions_learner_id",
        table_name="learner_quiz_sessions",
    )
    op.drop_table("learner_quiz_sessions")
