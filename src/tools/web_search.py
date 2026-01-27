import time
import random
import asyncio
from typing import List, Dict, Optional

from ddgs import DDGS
from httpx import ReadTimeout, ConnectTimeout


class DuckDuckGoSearch:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        timeout: float = 10.0,
    ):
        """
        :param max_retries: how many times to retry on failure
        :param base_delay: initial backoff delay (seconds)
        :param max_delay: maximum delay between retries
        :param timeout: request timeout (seconds)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Dict]:
        return await asyncio.to_thread(self._search_sync, query, max_results)

    def _search_sync(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Dict]:
        attempt = 0

        while attempt <= self.max_retries:
            try:
                with DDGS(timeout=self.timeout) as ddgs:
                    results = list(
                        ddgs.text(query, max_results=max_results)
                    )
                    return results

            except (ReadTimeout, ConnectTimeout, TimeoutError) as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise RuntimeError(
                        f"DuckDuckGo search failed after {self.max_retries} retries"
                    ) from e

                delay = min(
                    self.base_delay * (2 ** (attempt - 1)),
                    self.max_delay,
                )
                delay += random.uniform(0, 0.5)  # jitter
                time.sleep(delay)

            except Exception as e:
                # Non-network error → fail fast
                raise RuntimeError("Unexpected DDGS error") from e

        return []
