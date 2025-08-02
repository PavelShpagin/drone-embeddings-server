#!/usr/bin/env python3
"""Test remote AWS server."""

from local import test_both_modes_http

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Remote AWS Server')
    parser.add_argument('--host', type=str, default='ec2-16-171-238-14.eu-north-1.compute.amazonaws.com', 
                       help='AWS host (default: ec2-16-171-238-14.eu-north-1.compute.amazonaws.com)')
    parser.add_argument('--port', type=int, default=5000, help='Port (default: 5000)')
    parser.add_argument('--protocol', type=str, default='http', help='Protocol (default: http)')
    
    args = parser.parse_args()
    
    server_url = f"{args.protocol}://{args.host}:{args.port}"
    print(f"Testing AWS server at: {server_url}")
    
    results = test_both_modes_http(server_url)
    if results:
        print("AWS server test completed successfully")
    else:
        print("AWS server test failed")
