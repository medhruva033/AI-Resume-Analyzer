from app.database.connection import Base, engine

from app.models.user import User
from app.models.resume import Resume
from app.models.analysis import Analysis


def create_tables():
    Base.metadata.create_all(bind=engine)