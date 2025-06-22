#!/usr/bin/env python3
"""
Aegis Security Agent - Main Execution Script

This script initializes and runs the Aegis Security Agent using Google ADK.
It provides both interactive testing and API server capabilities.
"""

import asyncio
import sys
import time
from typing import Optional

# Add current directory to path for imports
# sys.path.insert(0, ".")
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.types import Content, Part

from aegis.aegis_agent import (
    create_aegis_agent,
    create_aegis_agent_with_callbacks,
)
from aegis.core.analyze_frame import init_analysis_engine
from aegis.core.security_context import initialize_global_context


class AegisSecuritySystem:
    """Main system class for running the Aegis Security Agent."""

    def __init__(self, use_callbacks: bool = True):
        """
        Initialize the Aegis Security System.

        Args:
            use_callbacks: Whether to enable security monitoring callbacks
        """
        self.agent = None
        self.runner = None
        self.session_service = None
        self.use_callbacks = use_callbacks

        # System configuration
        self.app_name = "aegis_security_system"
        self.default_user_id = "security_operator"
        self.default_session_id = "main_session"

    async def initialize(self) -> bool:
        """
        Initialize the security system components.

        Returns:
            bool: True if initialization successful, False otherwise
        """

        try:
            print("🛡️ INITIALIZING AEGIS SECURITY SYSTEM")
            print("=" * 50)

            # Initialize analysis engines (YOLO, VLM)
            print("🔧 Initializing AI analysis engines...")
            if init_analysis_engine():
                print("✅ AI analysis engines ready")
            else:
                print("⚠️ AI analysis engines partially initialized")

            # Create the security agent
            print("🤖 Creating Aegis Security Agent...")
            if self.use_callbacks:
                self.agent = create_aegis_agent_with_callbacks()
                print("✅ Agent created with security callbacks")
            else:
                self.agent = create_aegis_agent()
                print("✅ Agent created")

            # Initialize session service
            print("💾 Setting up session management...")
            self.session_service = InMemorySessionService()
            print("✅ Session service ready")

            # Create agent runner
            print("🚀 Initializing agent runner...")
            self.runner = Runner(
                agent=self.agent,
                app_name=self.app_name,
                session_service=self.session_service,
            )
            print("✅ Agent runner ready")

            # Create default session
            print("🔐 Creating default session...")
            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=self.default_user_id,
                session_id=self.default_session_id,
            )
            print("✅ Default session created")

            print("=" * 50)
            print("🎯 AEGIS SECURITY SYSTEM READY")
            print("📡 Available Tools:")
            for tool in self.agent.tools:
                tool_name = tool.__name__ if hasattr(tool, "__name__") else str(tool)
                print(f"   • {tool_name}")
            print("=" * 50)

            return True

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False

    async def process_command(
        self,
        command: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Process a security command through the agent.

        Args:
            command: Natural language security command
            user_id: User identifier (optional)
            session_id: Session identifier (optional)

        Returns:
            dict: Agent response with results
        """

        if not self.runner:
            return {
                "status": "error",
                "message": "System not initialized",
                "response": None,
            }

        try:
            user_id = user_id or self.default_user_id
            session_id = session_id or self.default_session_id

            print(f"\n🎙️ COMMAND: {command}")
            print(f"👤 User: {user_id} | 🔗 Session: {session_id}")
            print("-" * 40)

            start_time = time.time()

            content = types.Content(
                role="user", parts=[types.Part.from_text(text=command)]
            )
            
            # Run the agent
            event_stream = self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
            )

            final_response = "Agent did not produce a final response."
            async for event in event_stream:
                # Capture the last message from the agent
                if event.content.parts and event.content.parts[0].text:
                    text_parts = [
                        part.text
                        for part in event.content.parts
                        if hasattr(part, "text")
                    ]
                    if text_parts:
                        final_response = " ".join(text_parts)
                        # print(f"🤖 Aegis Response:\n{final_response}")

            processing_time = time.time() - start_time

            print(f"⏱️ Processing time: {processing_time:.2f}s")
            print("-" * 40)

            return {
                "status": "success",
                "command": command,
                "response": final_response,
                "processing_time": processing_time,
                "user_id": user_id,
                "session_id": session_id,
            }

        except Exception as e:
            print(f"❌ Command processing failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "command": command,
                "response": None,
            }

    async def run_interactive_mode(self):
        """Run the system in interactive command-line mode."""

        print("\n🎮 INTERACTIVE MODE")
        print("Type 'exit' to quit, 'help' for examples")
        print("=" * 50)

        while True:
            try:
                command = input("\n🛡️ Aegis> ").strip()

                if command.lower() in ["exit", "quit", "q"]:
                    print("👋 Shutting down Aegis Security System...")
                    break

                if command.lower() in ["help", "h"]:
                    self._show_help()
                    continue

                if not command:
                    continue

                # Process the command
                result = await self.process_command(command)

                if result["status"] == "success":
                    print(f"\n🤖 Aegis Response:\n{result['response']}")
                else:
                    print(f"\n❌ Error: {result['message']}")

            except KeyboardInterrupt:
                print("\n\n👋 Shutting down Aegis Security System...")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")

    def _show_help(self):
        """Show example commands and usage."""

        print("\n📖 AEGIS COMMAND EXAMPLES:")
        print("-" * 30)
        print("🎯 Object Detection:")
        print("   'Scan camera 3 for backpacks'")
        print("   'Look for weapons in gate 2'")
        print("   'Find any abandoned objects at parking area'")
        print()
        print("🔍 Scene Analysis:")
        print("   'Is the crowd dense at camera 1?'")
        print("   'Are people acting suspiciously at main entrance?'")
        print("   'What's happening at gate 4?'")
        print()
        print("🛡️ Security Assessment:")
        print("   'Analyze the security situation at cam2'")
        print("   'Check threat level at all gates'")
        print("   'Assess crowd control needs at plaza'")
        print()
        print("📹 Camera Control:")
        print("   'Switch to camera 4'")
        print("   'Show me the parking area feed'")
        print("   'List all available cameras'")
        print()
        print("📋 Incident Management:")
        print("   'Log security incident at gate 1'")
        print("   'Show recent incidents'")
        print("-" * 30)


async def main():
    """Main execution function."""

    print("🛡️ AEGIS - AI Security Co-Pilot System")
    print("Powered by Google ADK")
    print()

    # Create and initialize the system
    system = AegisSecuritySystem(use_callbacks=True)
    # initialize the global context
    context = initialize_global_context()
    if not context.is_initialized():
        print("❌ Failed to initialize SecurityContext")
        return


    if not await system.initialize():
        print("❌ Failed to initialize system")
        return 1

    # Check command line arguments
    if len(sys.argv) > 1:
        command = " ".join(sys.argv[1:])
        print(f"🎯 Executing command: {command}")

        result = await system.process_command(command)

        if result["status"] == "success":
            print(f"\n🤖 Aegis Response:\n{result['response']}")
            return 0
        else:
            print(f"\n❌ Error: {result['message']}")
            return 1
    else:
        # Run interactive mode
        await system.run_interactive_mode()
        return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
