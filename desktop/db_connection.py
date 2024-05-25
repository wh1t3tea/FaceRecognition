from typing import Annotated

from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from config import settings

engine = create_engine(
    url=settings.DATABASE_URL_psycopg,
    echo=True
)

session_factory = sessionmaker(engine)

str_256 = Annotated[str, 256]


class Base(DeclarativeBase):
    type_annotation_map = {
        str_256: String(256)
    }


def create_tables():
    engine.echo = False
    Base.metadata.create_all(engine)
    engine.echo = True

