"""Deep Translator (Google Translate) backend with timeout handling."""

from __future__ import annotations

import logging
import signal

from speakeasy.translation.base import Translator

logger = logging.getLogger(__name__)

# Mapping from Azure language codes to Google Translate codes.
_AZURE_TO_GOOGLE: dict[str, str] = {
    "zh-Hans": "zh-CN",
    "zh-Hant": "zh-TW",
}


def _timeout_handler(signum: int, frame: object) -> None:
    raise TimeoutError("Translation timed out")


class DeepTranslator(Translator):
    """Translator using the ``deep_translator`` library (Google Translate).

    Args:
        timeout: Maximum seconds to wait per translation call.
    """

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def _translate(self, text: str, source: str, target: str) -> str:
        from deep_translator import GoogleTranslator

        if not text:
            return ""

        source = _AZURE_TO_GOOGLE.get(source, source)
        target = _AZURE_TO_GOOGLE.get(target, target)

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(self.timeout)
        try:
            result = None
            for _ in range(5):
                result = GoogleTranslator(source=source, target=target).translate(text)
                if result is not None:
                    break
            return result or ""
        except TimeoutError:
            logger.warning("Translation timed out for: %s", text[:80])
            return ""
        except (TypeError, IndexError) as e:
            logger.error("Translation error (%s): %s", type(e).__name__, text[:80])
            return ""
        finally:
            signal.alarm(0)

    def translate_multilingual(self, sentence: str, source: str, target: str) -> str:
        return self._translate(sentence, source, target)

    def translate_to_english(
        self,
        sentence: str,
        source: str,
        target: str = "en",
        save_dir: str | None = None,
    ) -> str:
        return self._translate(sentence, source, target)
