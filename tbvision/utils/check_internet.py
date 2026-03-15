import socket


def check_internet_connection(timeout=3):
    """Check if internet connection is available"""

    try:
        # Try to connect to Google DNS
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        pass
    try:
        # Fallback: try Cloudflare DNS
        socket.create_connection(("1.1.1.1", 53), timeout=timeout)
        return True
    except OSError:
        return False
