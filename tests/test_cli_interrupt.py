from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path


class CliInterruptTests(unittest.TestCase):
    def test_rollback_interrupted_turn_removes_pending_user_message(self) -> None:
        from nova.cli.repl import _rollback_interrupted_turn
        from nova.session.store import SessionStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / ".nova" / "sessions", workspace=root)
            session_key = "session_test"
            store.save_state(
                session_key,
                {
                    "key": session_key,
                    "metadata": {
                        "pending_user_turn": True,
                        "runtime_checkpoint": {"phase": "awaiting_tools"},
                    },
                    "messages": [
                        {"role": "assistant", "content": "Earlier reply"},
                        {"role": "user", "content": "Interrupted question"},
                    ],
                },
            )

            history = [
                {"role": "assistant", "content": "Earlier reply"},
                {"role": "user", "content": "Interrupted question"},
            ]
            _rollback_interrupted_turn(store, session_key, history)

            state = store.load_state(session_key, restore_interrupted=False)
            self.assertNotIn("pending_user_turn", state["metadata"])
            self.assertNotIn("runtime_checkpoint", state["metadata"])
            self.assertEqual(state["messages"], [{"role": "assistant", "content": "Earlier reply"}])
            self.assertEqual(history, [{"role": "assistant", "content": "Earlier reply"}])

    def test_interrupt_controller_cancels_active_generation_task(self) -> None:
        from nova.cli.repl import _GenerationInterrupted, _InterruptController

        async def _run() -> str:
            controller = _InterruptController()

            async def slow_generation() -> str:
                await asyncio.sleep(60)
                return "done"

            task = asyncio.create_task(
                controller.run_interruptible(slow_generation())
            )
            await asyncio.sleep(0)
            controller._handle_sigint(None, None)
            with self.assertRaises(_GenerationInterrupted):
                await task
            self.assertIsNone(controller.active_task)
            return "ok"

        self.assertEqual(asyncio.run(_run()), "ok")


if __name__ == "__main__":
    unittest.main()
