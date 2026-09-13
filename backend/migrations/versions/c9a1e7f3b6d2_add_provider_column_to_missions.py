"""add provider column to missions

Revision ID: c9a1e7f3b6d2
Revises: 35932376d4e3
Create Date: 2026-08-23 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c9a1e7f3b6d2'
down_revision = '35932376d4e3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'missions',
        sa.Column('provider', sa.String(length=20), nullable=False, server_default='mistral'),
    )


def downgrade() -> None:
    op.drop_column('missions', 'provider')
