"""Central logging configuration.

Modules obtain their logger with the stdlib idiom ``logging.getLogger(__name__)``
at import time; :func:`configure_logging` (called once from the FastAPI lifespan)
installs the root handler and level. Stdlib logging is enough here — no
structured-logging dependency until log shipping actually requires it.
"""

import logging
import sys

from app.config import settings

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging() -> None:
    """
    Install the root logging handler and level from settings.

    Idempotent via ``force=True`` (reconfigures cleanly on reload).

    :return: Nothing.
    :rtype: None
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=_LOG_FORMAT, stream=sys.stdout, force=True)
