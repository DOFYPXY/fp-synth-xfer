import logging
from pathlib import Path
from typing import Any, Protocol, cast


class Logger(Protocol):
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def perf(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def config(self, msg: str, *args: Any, **kwargs: Any) -> None: ...


_PERF_LEVEL_NUM = 25
_CONFIG_LEVEL_NUM = 21

logging.addLevelName(_PERF_LEVEL_NUM, "PERF")
logging.addLevelName(_CONFIG_LEVEL_NUM, "CONFIG")


def _perf(self, msg, *args, **kwargs):
    if self.isEnabledFor(_PERF_LEVEL_NUM):
        self._log(_PERF_LEVEL_NUM, msg, args, **kwargs)


def _config(self, msg, *args, **kwargs):
    if self.isEnabledFor(_CONFIG_LEVEL_NUM):
        self._log(_CONFIG_LEVEL_NUM, msg, args, **kwargs)


logging.Logger.perf = _perf  # type: ignore[attr-defined]
logging.Logger.config = _config  # type: ignore[attr-defined]


class _ExactLevelFilter(logging.Filter):
    def __init__(self, level):
        super(_ExactLevelFilter, self).__init__()
        self.level = level

    def filter(self, record):
        return record.levelno == self.level


class _ExcludeLevelsFilter(logging.Filter):
    def __init__(self, *levels):
        super(_ExcludeLevelsFilter, self).__init__()
        self.levels = set(levels)

    def filter(self, record):
        return record.levelno not in self.levels


_LOGGER: Logger | None = None
_LOG_DIR: Path | None = None


def init_logging(log_dir: Path, verbose: bool) -> Logger:
    global _LOGGER, _LOG_DIR

    logger = logging.getLogger(f"custom_logger_{log_dir}")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(message)s")

    info_fh = logging.FileHandler(log_dir.joinpath("info.log"), mode="w")
    info_fh.setLevel(logging.INFO)
    info_fh.setFormatter(fmt)
    info_fh.addFilter(_ExcludeLevelsFilter(_PERF_LEVEL_NUM, _CONFIG_LEVEL_NUM))
    logger.addHandler(info_fh)

    # TODO use a log stream instead of printing
    # info_console = logging.StreamHandler()
    # info_console.setLevel(logging.INFO)
    # info_console.setFormatter(fmt)
    # info_console.addFilter(_ExcludeLevelsFilter(_PERF_LEVEL_NUM, _CONFIG_LEVEL_NUM))
    # logger.addHandler(info_console)

    if verbose:
        debug_fh = logging.FileHandler(log_dir.joinpath("debug.log"), mode="w")
        debug_fh.setLevel(logging.DEBUG)
        debug_fh.setFormatter(fmt)
        debug_fh.addFilter(_ExcludeLevelsFilter(_PERF_LEVEL_NUM, _CONFIG_LEVEL_NUM))
        logger.addHandler(debug_fh)

    perf_fh = logging.FileHandler(log_dir.joinpath("perf.log"), mode="w")
    perf_fh.setLevel(_PERF_LEVEL_NUM)
    perf_fh.setFormatter(fmt)
    perf_fh.addFilter(_ExactLevelFilter(_PERF_LEVEL_NUM))
    logger.addHandler(perf_fh)

    config_fh = logging.FileHandler(log_dir.joinpath("config.log"), mode="w")
    config_fh.setLevel(_CONFIG_LEVEL_NUM)
    config_fh.setFormatter(fmt)
    config_fh.addFilter(_ExactLevelFilter(_CONFIG_LEVEL_NUM))
    logger.addHandler(config_fh)

    _LOG_DIR = log_dir
    _LOGGER = cast(Logger, logger)
    return _LOGGER


def get_logger() -> Logger:
    global _LOGGER
    if _LOGGER is None:
        raise RuntimeError("init_logging() must be called first.")
    return _LOGGER


def write_log_file(filename: str, contents: Any) -> Path:
    if _LOG_DIR is None:
        raise RuntimeError("init_logging() must be called first.")

    path = _LOG_DIR.joinpath(filename)
    path.write_text(str(contents))
    return path
