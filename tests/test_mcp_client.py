from __future__ import annotations

import re
import unittest


class MCPClientTests(unittest.TestCase):
    def test_sanitize_tool_name_for_openai_function_schema(self) -> None:
        from nova.mcp.client import _sanitize_tool_name

        name = _sanitize_tool_name("mcp_example-server_resource_test://static resource/1")

        self.assertRegex(name, re.compile(r"^[A-Za-z0-9_-]{1,64}$"))
        self.assertNotIn(":", name)
        self.assertNotIn("/", name)
        self.assertNotIn(" ", name)

    def test_reserve_tool_name_deduplicates_collisions(self) -> None:
        from nova.mcp.client import MCPClient

        client = MCPClient({})

        first = client._reserve_tool_name("mcp_server_tool")
        second = client._reserve_tool_name("mcp_server_tool")

        self.assertEqual(first, "mcp_server_tool")
        self.assertEqual(second, "mcp_server_tool_2")


if __name__ == "__main__":
    unittest.main()
