class GameException(Exception):
    def __init__(self, message: str | None = None):
        super().__init__(message)
        self._message = message

    @property
    def message(self) -> str | None:
        return self._message
    
