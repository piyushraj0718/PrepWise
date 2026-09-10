"""Add bounded question generation attempt audit records."""
from alembic import op
import sqlalchemy as sa


revision = "0002_assessment_question_attempts"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assessment_question_attempts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "generation_run_id",
            sa.String(36),
            sa.ForeignKey("assessment_generation_runs.id"),
            nullable=False,
        ),
        sa.Column("question_slot", sa.Integer(), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("candidate_payload", sa.JSON(), nullable=True),
        sa.Column("failure_reasons", sa.JSON(), nullable=True),
        sa.Column(
            "accepted_question_id",
            sa.String(36),
            sa.ForeignKey("assessment_questions.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "generation_run_id",
            "question_slot",
            "attempt_number",
            name="uq_assessment_question_attempt",
        ),
    )
    op.create_index(
        "ix_assessment_question_attempts_generation_run_id",
        "assessment_question_attempts",
        ["generation_run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_assessment_question_attempts_generation_run_id",
        table_name="assessment_question_attempts",
    )
    op.drop_table("assessment_question_attempts")
