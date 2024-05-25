from sqlalchemy import select

from tables import SubscriptionOrm
from db_connection import session_factory


class Authorization:
    """
    Class for user authorization.
    """

    def __init__(self, api_key):
        """
        Initialize the Authorization object.

        Args:
            api_key (str): User's API key.
        """
        assert isinstance(api_key, str)
        self.api_key = api_key

    def login(self):
        """
        Login method to authenticate the user.

        Returns:
            bool: True if login is successful, False otherwise.
        """
        with session_factory() as session:
            try:
                query = select(SubscriptionOrm.app_key).where(
                    SubscriptionOrm.app_key == self.api_key,
                    SubscriptionOrm.subscription_plan != "expired",
                )
                result = session.execute(query).all()

                return len(result) > 0
            except Exception as e:
                print(f"Ошибка при выполнении входа в систему: {e}")
                return False
