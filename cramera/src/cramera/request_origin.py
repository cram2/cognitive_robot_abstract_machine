"""Browser-origin validation for the local viewer and live bridge."""

from dataclasses import dataclass
from enum import StrEnum
from ipaddress import ip_address
from urllib.parse import urlsplit


# %% origin vocabulary


class OriginHeader(StrEnum):
    """HTTP headers controlling browser access to a local server."""

    ORIGIN = "Origin"
    """The browser page submitting the request."""

    ALLOW_ORIGIN = "Access-Control-Allow-Origin"
    """The browser origin allowed to read this response."""

    VARY = "Vary"
    """The request headers affecting the response."""


@dataclass(frozen=True)
class RequestOrigin:
    """The browser origin presented to a local HTTP server."""

    value: str | None
    """The Origin header, absent for local scripts and command-line clients."""

    def is_allowed(self) -> bool:
        """Allow local clients and pages served from a loopback address."""
        if self.value is None:
            return True
        try:
            origin = urlsplit(self.value)
            origin.port
        except ValueError:
            return False
        if (
            origin.scheme not in {"http", "https"}
            or not origin.hostname
            or origin.username is not None
            or origin.path
            or origin.query
            or origin.fragment
        ):
            return False
        if origin.hostname == "localhost":
            return True
        try:
            return ip_address(origin.hostname).is_loopback
        except ValueError:
            return False

    def response_value(self) -> str | None:
        """Return the trusted origin to reflect, or no permission when rejected."""
        if not self.is_allowed():
            return None
        return self.value if self.value is not None else "*"
