import aiohttp
import logging

from typing import Any

class ApiClient:
    def __init__(self, base_url:str) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=20)
        self._session: aiohttp.ClientSession | None = None
        self._logger = logging.getLogger(__name__)
        pass

    async def __aenter__(self):
        self._session: aiohttp.ClientSession = aiohttp.ClientSession(base_url = self._base_url, timeout = self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def post(self, path: str, *, json: dict[str, Any] | None = None) -> aiohttp.ClientResponse:       
        if not self._session:
            raise RuntimeError("Session not initialized; use 'async with ApiClient(...)'")
        url = f"{self._base_url}{path}"
        return await self._session.post(url, json=json)
    
    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.post(path, json=payload)
        text = await response.text()                      
        if response.status >= 400:
            self._logger.error(
                "HTTP %s %s body=%s", response.status, response.request_info, text
            )
            raise aiohttp.ClientResponseError(
                request_info=response.request_info,
                history=response.history,
                status=response.status,
                message=text,                             
                headers=response.headers,
            )
        return await response.json()
