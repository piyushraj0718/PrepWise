"""Add immutable assessment quality evaluation audit records."""
from alembic import op
import sqlalchemy as sa

revision = "0003_assessment_quality_evaluations"
down_revision = "0002_assessment_question_attempts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assessment_quality_evaluations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("generation_run_id", sa.String(36), sa.ForeignKey("assessment_generation_runs.id"), nullable=False),
        sa.Column("question_attempt_id", sa.String(36), sa.ForeignKey("assessment_question_attempts.id"), nullable=False),
        sa.Column("question_id", sa.String(36), sa.ForeignKey("assessment_questions.id"), nullable=True),
        sa.Column("evaluation_type", sa.String(30), nullable=False),
        sa.Column("evaluator_provider", sa.String(100), nullable=True),
        sa.Column("evaluator_model", sa.String(255), nullable=True),
        sa.Column("prompt_version", sa.String(100), nullable=True),
        sa.Column("overall_score", sa.Float(), nullable=True),
        sa.Column("dimension_scores", sa.JSON(), nullable=True),
        sa.Column("recommendation", sa.String(20), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("question_attempt_id", "evaluation_type", name="uq_attempt_quality_type"),
    )
    for column in ("generation_run_id", "question_attempt_id", "question_id"):
        op.create_index(f"ix_assessment_quality_evaluations_{column}", "assessment_quality_evaluations", [column])


def downgrade() -> None:
    for column in ("question_id", "question_attempt_id", "generation_run_id"):
        op.drop_index(f"ix_assessment_quality_evaluations_{column}", table_name="assessment_quality_evaluations")
    op.drop_table("assessment_quality_evaluations")
