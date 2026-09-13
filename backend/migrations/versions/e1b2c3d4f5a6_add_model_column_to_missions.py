"""add model column to missions

Revision ID: e1b2c3d4f5a6
Revises: c9a1e7f3b6d2
Create Date: 2026-09-13 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e1b2c3d4f5a6'
down_revision = 'c9a1e7f3b6d2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'missions',
        sa.Column('model', sa.String(length=80), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('missions', 'model')
