#!/bin/bash

# fetch.sh - Fetch logs from drone embeddings server
# Usage: ./fetch.sh [session_id] [logger_id] [server_url]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEFAULT_SERVER="http://ec2-16-171-238-14.eu-north-1.compute.amazonaws.com:5000"
LOGS_DIR="data/logs"

# Functions
print_usage() {
    echo "Usage: $0 [OPTIONS] [SESSION_ID] [LOGGER_ID]"
    echo ""
    echo "OPTIONS:"
    echo "  -s, --server URL     Server URL (default: AWS server)"
    echo "  -o, --output DIR     Output directory (default: $LOGS_DIR)"
    echo "  -l, --list          List available sessions"
    echo "  -i, --info SESSION  Get info about a specific session"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 --list                           # List all sessions"
    echo "  $0 --info abc123                    # Get session info"
    echo "  $0 abc123                           # Download all logs"
    echo "  $0 abc123 logger456                 # Download specific logger"
    echo "  $0 -s http://localhost:5000 abc123  # Use local server"
    echo ""
    echo "Default server: $DEFAULT_SERVER"
}

print_error() {
    echo -e "${RED}Error: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Parse command line arguments
SERVER_URL="$DEFAULT_SERVER"
OUTPUT_DIR="$LOGS_DIR"
LIST_SESSIONS=false
SESSION_INFO=""
SESSION_ID=""
LOGGER_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--server)
            SERVER_URL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -l|--list)
            LIST_SESSIONS=true
            shift
            ;;
        -i|--info)
            SESSION_INFO="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        -*)
            print_error "Unknown option $1"
            print_usage
            exit 1
            ;;
        *)
            if [[ -z "$SESSION_ID" ]]; then
                SESSION_ID="$1"
            elif [[ -z "$LOGGER_ID" ]]; then
                LOGGER_ID="$1"
            else
                print_error "Too many arguments"
                print_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if curl is available
if ! command -v curl &> /dev/null; then
    print_error "curl is required but not installed"
    exit 1
fi

# Check if jq is available for JSON parsing
if ! command -v jq &> /dev/null; then
    print_warning "jq not found - JSON output will be raw"
    JQ_AVAILABLE=false
else
    JQ_AVAILABLE=true
fi

# Function to make API calls
api_call() {
    local endpoint="$1"
    local method="${2:-GET}"
    local data="$3"
    
    local curl_cmd="curl -s"
    
    if [[ "$method" == "POST" ]]; then
        curl_cmd="$curl_cmd -X POST -H 'Content-Type: application/json'"
        if [[ -n "$data" ]]; then
            curl_cmd="$curl_cmd -d '$data'"
        fi
    fi
    
    curl_cmd="$curl_cmd '$SERVER_URL/$endpoint'"
    
    eval "$curl_cmd"
}

# Function to list available sessions
list_available_sessions() {
    print_info "Fetching available sessions from $SERVER_URL..."
    
    response=$(api_call "available_logs")
    
    if [[ $JQ_AVAILABLE == true ]]; then
        echo "$response" | jq -r '
            if .success then
                "Available Sessions (" + (.total_sessions | tostring) + " total):",
                "=" * 50,
                (.sessions[] | 
                    "Session: " + .session_id,
                    "  Loggers: " + (.logger_count | tostring),
                    (.loggers[] | "    - " + .logger_id + " (" + (.file_count | tostring) + " files)"),
                    ""
                )
            else
                "Error: " + (.error // "Unknown error")
            end
        '
    else
        echo "$response"
    fi
}

# Function to get session info
get_session_info() {
    local session_id="$1"
    print_info "Fetching info for session $session_id..."
    
    response=$(api_call "logs_summary/$session_id")
    
    if [[ $JQ_AVAILABLE == true ]]; then
        echo "$response" | jq -r '
            if .success then
                "Session: " + .session_id,
                "Total Files: " + (.total_files | tostring),
                "Total Size: " + ((.total_size / 1024 / 1024 * 100 | floor) / 100 | tostring) + " MB",
                "Loggers:",
                (.loggers | to_entries[] | 
                    "  " + .key + ":",
                    "    Files: " + (.value.file_count | tostring),
                    "    Size: " + ((.value.total_size / 1024 | floor) | tostring) + " KB",
                    (.value.files[] | "      - " + .filename + " (" + ((.size / 1024 | floor) | tostring) + " KB)")
                )
            else
                "Error: " + (.message // .error // "Unknown error")
            end
        '
    else
        echo "$response"
    fi
}

# Function to download logs
download_logs() {
    local session_id="$1"
    local logger_id="$2"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Prepare filename and API data
    if [[ -n "$logger_id" ]]; then
        filename="logs_${session_id}_${logger_id}.zip"
        api_data="{\"session_id\": \"$session_id\", \"logger_id\": \"$logger_id\"}"
        print_info "Downloading logs for session $session_id, logger $logger_id..."
    else
        filename="logs_${session_id}_all.zip"
        api_data="{\"session_id\": \"$session_id\"}"
        print_info "Downloading all logs for session $session_id..."
    fi
    
    # Download the file and check if it's a zip
    print_info "Downloading logs..."
    
    # Create a temporary file to store the response
    temp_file=$(mktemp)
    
    # Download the response and capture HTTP status
    http_status=$(curl -s -w "%{http_code}" -X POST -H 'Content-Type: application/json' \
                      -d "$api_data" "$SERVER_URL/fetch_logs" -o "$temp_file")
    
    if [[ "$http_status" == "200" ]]; then
        # Check if the response is a zip file by looking at the first few bytes
        if file "$temp_file" | grep -q "Zip archive"; then
            # It's a zip file, move it to the final location
            mv "$temp_file" "$OUTPUT_DIR/$filename"
            file_size=$(ls -lh "$OUTPUT_DIR/$filename" | awk '{print $5}')
            print_success "Downloaded: $OUTPUT_DIR/$filename ($file_size)"
            
            # Ask if user wants to extract
            echo -n "Extract the zip file? [y/N]: "
            read -r extract_choice
            
            if [[ "$extract_choice" =~ ^[Yy]$ ]]; then
                extract_dir="$OUTPUT_DIR/$(basename "$filename" .zip)"
                mkdir -p "$extract_dir"
                
                if command -v unzip &> /dev/null; then
                    unzip -q "$OUTPUT_DIR/$filename" -d "$extract_dir"
                    print_success "Extracted to: $extract_dir"
                    
                    # Show contents
                    print_info "Contents:"
                    find "$extract_dir" -type f | head -20 | while read -r file; do
                        echo "  - $(basename "$file")"
                    done
                    
                    if [[ $(find "$extract_dir" -type f | wc -l) -gt 20 ]]; then
                        echo "  ... and $(($(find "$extract_dir" -type f | wc -l) - 20)) more files"
                    fi
                else
                    print_warning "unzip not available - manual extraction required"
                fi
            fi
        else
            # It's probably an error response (JSON)
            error_response=$(cat "$temp_file")
            rm -f "$temp_file"
            
            print_error "Server returned an error:"
            if [[ $JQ_AVAILABLE == true ]]; then
                echo "$error_response" | jq -r '.error // .message // "Unknown error"'
            else
                echo "$error_response"
            fi
            exit 1
        fi
    else
        # HTTP error
        error_response=$(cat "$temp_file")
        rm -f "$temp_file"
        
        print_error "HTTP $http_status error:"
        if [[ $JQ_AVAILABLE == true ]]; then
            echo "$error_response" | jq -r '.error // .message // "HTTP error"'
        else
            echo "$error_response"
        fi
        exit 1
    fi
    
    # Clean up temp file if it still exists
    rm -f "$temp_file"
}

# Main logic
if [[ "$LIST_SESSIONS" == true ]]; then
    list_available_sessions
elif [[ -n "$SESSION_INFO" ]]; then
    get_session_info "$SESSION_INFO"
elif [[ -n "$SESSION_ID" ]]; then
    download_logs "$SESSION_ID" "$LOGGER_ID"
else
    print_error "No action specified"
    print_usage
    exit 1
fi
