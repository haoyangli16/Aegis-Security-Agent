"""
Security Context for Aegis Security Agent System.

This module provides the SecurityContext class that holds shared resources
like the WebMonitorServer instance, ensuring efficient resource management
across all security tools.
"""

import logging
from typing import Optional

from aegis.server.web_monitor_server import WebMonitorServer

logger = logging.getLogger(__name__)


class SecurityContext:
    """
    Application context that holds shared resources for the Aegis Security System.
    
    This context ensures that resources like the WebMonitorServer are initialized
    once and shared across all tools, preventing resource duplication and improving
    performance.
    """
    
    def __init__(self):
        """Initialize the security context with shared resources."""
        self._monitor_server: Optional[WebMonitorServer] = None
        self._initialized = False
        logger.info("SecurityContext created")
    
    @property
    def monitor_server(self) -> WebMonitorServer:
        """
        Get the shared WebMonitorServer instance.
        
        Returns:
            WebMonitorServer: The shared monitor server instance
            
        Raises:
            RuntimeError: If the context is not properly initialized
        """
        if self._monitor_server is None:
            raise RuntimeError("SecurityContext not initialized. Call initialize() first.")
        return self._monitor_server
    
    def initialize(self) -> bool:
        """
        Initialize all shared resources.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if self._initialized:
                logger.info("SecurityContext already initialized")
                return True
            
            logger.info("Initializing SecurityContext...")
            
            # Initialize WebMonitorServer
            self._monitor_server = WebMonitorServer()
            logger.info("✅ WebMonitorServer initialized in SecurityContext")
            
            # Add other shared resources here in the future:
            # self._database_connection = DatabaseConnection()
            # self._config_manager = ConfigManager()
            # etc.
            
            self._initialized = True
            logger.info("✅ SecurityContext fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SecurityContext: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """Check if the context is properly initialized."""
        return self._initialized and self._monitor_server is not None
    
    def cleanup(self):
        """Clean up resources when shutting down."""
        try:
            if self._monitor_server:
                logger.info("Cleaning up WebMonitorServer...")
                self._monitor_server.stop_monitoring()
                self._monitor_server = None
            
            self._initialized = False
            logger.info("✅ SecurityContext cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Error during SecurityContext cleanup: {e}")


# Global context instance - will be initialized by the main server
_global_context: Optional[SecurityContext] = None


def get_security_context() -> SecurityContext:
    """
    Get the global SecurityContext instance.
    
    Returns:
        SecurityContext: The global security context
        
    Raises:
        RuntimeError: If the global context is not initialized
    """
    global _global_context
    if _global_context is None:
        raise RuntimeError("Global SecurityContext not initialized. Call initialize_global_context() first.")
    return _global_context


def initialize_global_context() -> SecurityContext:
    """
    Initialize the global SecurityContext instance.
    
    Returns:
        SecurityContext: The initialized global security context
    """
    global _global_context
    if _global_context is None:
        _global_context = SecurityContext()
        _global_context.initialize()
    return _global_context


def cleanup_global_context():
    """Clean up the global SecurityContext instance."""
    global _global_context
    if _global_context:
        _global_context.cleanup()
        _global_context = None 