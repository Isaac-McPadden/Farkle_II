# src/farkle/app_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

from farkle.analysis.analysis_config import PipelineCfg


@dataclass
class AppConfig:
    """Application-wide configuration container.

    Currently this only exposes the :class:`~farkle.analysis.analysis_config.PipelineCfg`
    used by the analysis pipeline but is designed to grow additional
    sections over time.
    """

    analysis: PipelineCfg = field(default_factory=PipelineCfg)

    @classmethod
    def parse_cli(cls, argv: Sequence[str] | None = None) -> Tuple["AppConfig", object, list[str]]:
        """Parse command line options into an :class:`AppConfig`.

        This simply forwards to :meth:`PipelineCfg.parse_cli` and wraps the
        resulting configuration inside :class:`AppConfig`.
        """

        cfg, ns, remaining = PipelineCfg.parse_cli(argv)
        return cls(analysis=cfg), ns, remaining
