"""Inkbox SMS proof-of-life notifications for long-running gateway turns."""

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class CaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.INKBOX):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id=f"msg-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class CaptureEditableAdapter(CaptureAdapter):
    async def edit_message(
        self,
        chat_id,
        message_id,
        content,
        *,
        finalize=False,
    ) -> SendResult:
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "metadata": {"edited": True, "finalize": finalize},
        })
        return SendResult(success=True, message_id=message_id)


class SlowAgent:
    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        time.sleep(0.15)
        return {"final_response": "final answer", "messages": [], "api_calls": 1}


class FastAgent:
    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        return {"final_response": "final answer", "messages": [], "api_calls": 1}


class ChatteryAgent:
    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = None
        self.interim_assistant_callback = None

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.tool_progress_callback:
            self.tool_progress_callback(
                "tool.started",
                tool_name="web_search",
                preview="private local plans query",
                args={"query": "private local plans query"},
            )
        if self.interim_assistant_callback:
            self.interim_assistant_callback("Calling web_search now.")
        time.sleep(0.15)
        return {"final_response": "final answer", "messages": [], "api_calls": 1}


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


def _install_fakes(monkeypatch, tmp_path, agent_cls, *, display=None):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    monkeypatch.setenv("HERMES_SMS_PROGRESS_INITIAL_SECONDS", "0.05")
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "fake"},
    )
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": display if display is not None else {
            "tool_progress": "off",
            "interim_assistant_messages": False,
        }},
    )


@pytest.mark.asyncio
async def test_inkbox_sms_long_turn_sends_human_safe_progress_ping(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, tmp_path, SlowAgent)
    adapter = CaptureAdapter(Platform.INKBOX)
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.INKBOX,
        chat_id="contact-123",
        chat_type="dm",
        user_id_alt="+15551234567",
    )

    result = await runner._run_agent(
        message="please research this",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-sms-progress",
        session_key="agent:main:inkbox:dm:contact-123",
    )

    assert result["final_response"] == "final answer"
    assert adapter.sent == [{
        "chat_id": "contact-123",
        "content": "I am checking that now.",
        "metadata": {"mode": "sms"},
    }]


@pytest.mark.asyncio
async def test_inkbox_sms_quick_turn_does_not_send_progress_ping(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, tmp_path, FastAgent)
    adapter = CaptureAdapter(Platform.INKBOX)
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.INKBOX,
        chat_id="contact-123",
        chat_type="dm",
        user_id_alt="+15551234567",
    )

    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-sms-fast",
        session_key="agent:main:inkbox:dm:contact-123",
    )

    assert result["final_response"] == "final answer"
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_inkbox_sms_pending_followup_suppresses_progress_ping(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, tmp_path, SlowAgent)
    adapter = CaptureAdapter(Platform.INKBOX)
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.INKBOX,
        chat_id="contact-123",
        chat_type="dm",
        user_id_alt="+15551234567",
    )
    session_key = "agent:main:inkbox:dm:contact-123"
    pending = asyncio.Event()
    pending.set()
    adapter._active_sessions[session_key] = pending

    result = await runner._run_agent(
        message="please research this",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-sms-pending",
        session_key=session_key,
    )

    assert result["final_response"] == "final answer"
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_inkbox_sms_suppresses_tool_progress_and_interim_chatter(monkeypatch, tmp_path):
    _install_fakes(
        monkeypatch,
        tmp_path,
        ChatteryAgent,
        display={"tool_progress": "all", "interim_assistant_messages": True},
    )
    adapter = CaptureEditableAdapter(Platform.INKBOX)
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.INKBOX,
        chat_id="contact-123",
        chat_type="dm",
        user_id_alt="+15551234567",
    )

    result = await runner._run_agent(
        message="please research this",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-sms-no-tool-chatter",
        session_key="agent:main:inkbox:dm:contact-123",
    )

    assert result["final_response"] == "final answer"
    assert adapter.sent == [{
        "chat_id": "contact-123",
        "content": "I am checking that now.",
        "metadata": {"mode": "sms"},
    }]


@pytest.mark.asyncio
async def test_inkbox_voice_call_does_not_use_sms_progress_ping(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, tmp_path, SlowAgent)
    adapter = CaptureAdapter(Platform.INKBOX)
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.INKBOX,
        chat_id="contact-123",
        chat_type="dm",
        user_id_alt="+15551234567",
        thread_id="call:call-123",
        chat_topic="voice_call",
    )

    result = await runner._run_agent(
        message="call transcript",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-voice-progress",
        session_key="agent:main:inkbox:dm:contact-123:call:call-123",
    )

    assert result["final_response"] == "final answer"
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_non_sms_turn_keeps_existing_long_running_notice(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, tmp_path, SlowAgent)
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.05")
    adapter = CaptureAdapter(Platform.TELEGRAM)
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")

    result = await runner._run_agent(
        message="please research this",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-telegram-progress",
        session_key="agent:main:telegram:dm:42",
    )

    assert result["final_response"] == "final answer"
    assert len(adapter.sent) >= 1
    assert "Still working..." in adapter.sent[0]["content"]
