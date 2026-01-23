import abc
import logging

logger = logging.getLogger("giskardpy")


class MiddlewareWrapper(abc.ABC):

    @abc.abstractmethod
    def loginfo(self, msg: str): ...

    @abc.abstractmethod
    def logwarn(self, msg: str): ...

    @abc.abstractmethod
    def logerr(self, msg: str): ...

    @abc.abstractmethod
    def logdebug(self, msg: str): ...

    @abc.abstractmethod
    def logfatal(self, msg: str): ...

    @abc.abstractmethod
    def resolve_iri(cls, path: str) -> str: ...


class NoMiddleware(MiddlewareWrapper):

    def loginfo(self, msg: str):
        logger.info(f"[INFO]: {msg}")

    def logwarn(self, msg: str):
        logger.warning(f"[WARN]: {msg}")

    def logerr(self, msg: str):
        logger.error(f"[ERROR]: {msg}")

    def logdebug(self, msg: str):
        logger.debug(f"[DEBUG]: {msg}")

    def logfatal(self, msg: str):
        logger.fatal(f"[FATAL]: {msg}")

    def resolve_iri(cls, path: str) -> str:
        return path
