"""
The rules on version control.
"""

from __future__ import annotations


from krrood.development_tools.agents.rule import Section


class VersionControlRules(Section):
    """
    The rules on commits, pushes and pull requests.
    """

    AUTHOR_AS_THE_HUMAN = (
        "Author commits as the human user running the tool, with their own "
        "configured git `user.name` and `user.email`. Never author or amend a "
        "commit as an assistant, and never credit one as author or co-author: no "
        "`Co-Authored-By:` trailer for an assistant, no `noreply@anthropic.com` or "
        "similar as author or committer"
    )

    ACKNOWLEDGE_ASSISTANCE_PLAINLY = (
        "A short plain line in the commit message body acknowledging assistant "
        "help, such as `Made with AI assistance`, is encouraged. Never name the "
        "model or the AI service"
    )

    RULES_APPLY_TO_EVERYONE = "These rules apply to every contributor and every tool"

    NEVER_PUSH_UPSTREAM = (
        "When you work in a fork, never push to the upstream repository - no "
        "branch, no tag, whatever the reason. Push only to the fork; work reaches "
        "upstream only through a pull request opened from the fork. Before any "
        "push, read the full, untruncated `git remote -v` output and confirm the "
        "target remote is the fork"
    )

    LEAVE_UPSTREAM_PULL_REQUESTS_ALONE = (
        "Never comment on or modify pull requests on the upstream repository "
        "unless the user has explicitly allowed it - through existing personal "
        "notes or instructions, or by accepting when you ask"
    )
