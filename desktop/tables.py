import datetime
from typing import Annotated
from sqlalchemy.orm import mapped_column
from db_connection import Base


from sqlalchemy.orm import Mapped

intpk = Annotated[int, mapped_column(primary_key=True)]


class SubscriptionOrm(Base):
    __tablename__ = "core_subscription"

    user_id: Mapped[intpk]
    subscription_plan: Mapped[str]
    valid_until: Mapped[datetime.datetime]
    app_key: Mapped[str]


