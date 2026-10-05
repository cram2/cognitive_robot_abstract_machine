"""
A GitHub repository, identified the way GitHub itself writes it, and read out of the
references and git remote URLs the tooling is handed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MalformedRepositoryError(ValueError):
    """
    Raised when a repository reference is not in ``owner/name`` form.
    """

    text: str
    """
    The value that could not be parsed.
    """

    def __str__(self) -> str:
        """:return: What was expected and what arrived instead."""
        return f"expected a repository as 'owner/name', got {self.text!r}"


@dataclass(frozen=True)
class Repository:
    """
    A GitHub repository, identified the way GitHub itself writes it.
    """

    owner: str
    """
    The user or organization the repository belongs to.
    """

    name: str
    """
    The repository's own name.
    """

    @classmethod
    def parse(cls, text: str) -> Repository:
        """
        Parse an ``owner/name`` repository reference.

        :param text: The reference to parse.
        :return: The parsed repository.
        :raises MalformedRepositoryError: If *text* is not ``owner/name``.
        """
        owner, separator, name = text.partition("/")
        if not (owner and separator and name):
            raise MalformedRepositoryError(text)
        return cls(owner, name)

    @staticmethod
    def _remote_url_segments(url: str) -> list[str]:
        """
        Split a remote URL into its path segments, discarding scheme and host.

        :param url: The remote URL to split.
        :return: The path segments, which name a repository when there are two or more.
        """
        reference = url.removesuffix(".git").rstrip("/")
        if "://" in reference:
            _, _, host_and_path = reference.partition("://")
            _, _, path = host_and_path.partition("/")
        elif ":" in reference:
            _, _, path = reference.rpartition(":")
        else:
            return []
        return [segment for segment in path.split("/") if segment]

    @classmethod
    def names_a_repository(cls, url: str) -> bool:
        """
        Test whether a remote URL points at a repository at all.

        :param url: The remote URL to test.
        :return: Whether it names an ``owner/name`` pair.
        """
        return len(cls._remote_url_segments(url)) >= 2

    @classmethod
    def from_remote_url(cls, url: str) -> Repository:
        """
        Read the repository a git remote URL points at.

        Accepts every form a fork remote takes - HTTPS, SSH, and the local proxy a cloud
        session is given - by discarding the host and taking the last two path segments.

        :param url: The remote URL to read.
        :return: The repository it names.
        :raises MalformedRepositoryError: If *url* names no ``owner/name`` pair.
        """
        segments = cls._remote_url_segments(url)
        if len(segments) < 2:
            raise MalformedRepositoryError(url)
        return cls.parse("/".join(segments[-2:]))

    @property
    def full_name(self) -> str:
        """
        The ``owner/name`` form GitHub's own interface and the ``gh`` CLI use.
        """
        return f"{self.owner}/{self.name}"

    def __str__(self) -> str:
        """:return: :attr:`full_name`."""
        return self.full_name
