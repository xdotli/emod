#!/bin/bash
# EMOD Experiment Results Analysis
# This script runs the full results processing workflow

# Print colored text
print_green() {
    echo -e "\033[0;32m$1\033[0m"
}

print_blue() {
    echo -e "\033[0;34m$1\033[0m"
}

print_red() {
    echo -e "\033[0;31m$1\033[0m"
}

print_yellow() {
    echo -e "\033[0;33m$1\033[0m"
}

# Print header
print_blue "=================================================="
print_blue "       EMOD EXPERIMENT RESULTS ANALYSIS           "
print_blue "=================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_red "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if script files exist
for script in process_all_results.py download_modal_results.py enhanced_logging.py generate_report.py; do
    if [ ! -f "$script" ]; then
        print_red "Error: Required script $script not found"
        print_yellow "Make sure you're running this from the project root directory"
        exit 1
    fi
done

# Parse arguments
SKIP_DOWNLOAD=0
LIST_ONLY=0
DOWNLOAD_ONLY=0
TARGET_DIR="./results"

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=1
            shift
            ;;
        --list-only)
            LIST_ONLY=1
            shift
            ;;
        --download-only)
            DOWNLOAD_ONLY=1
            shift
            ;;
        --target-dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        --help)
            print_blue "Usage: ./run_results_analysis.sh [OPTIONS]"
            print_blue "Options:"
            print_blue "  --skip-download   Skip downloading results from Modal"
            print_blue "  --list-only       Only list Modal volume contents without downloading"
            print_blue "  --download-only   Only download results without processing"
            print_blue "  --target-dir DIR  Set target directory for results (default: ./results)"
            print_blue "  --help            Show this help message"
            exit 0
            ;;
        *)
            print_red "Unknown option: $1"
            print_yellow "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Run the main processing script with appropriate arguments
ARGS=""
if [ $SKIP_DOWNLOAD -eq 1 ]; then
    ARGS="$ARGS --skip-download"
fi
if [ $LIST_ONLY -eq 1 ]; then
    ARGS="$ARGS --list-only"
fi
if [ $DOWNLOAD_ONLY -eq 1 ]; then
    ARGS="$ARGS --download-only"
fi
if [ "$TARGET_DIR" != "./results" ]; then
    ARGS="$ARGS --target-dir $TARGET_DIR"
fi

print_green "Starting results processing..."
python3 process_all_results.py $ARGS

# Check exit status
if [ $? -eq 0 ]; then
    print_green "\nResults processing completed successfully!"
    
    # Show report location if processing was done
    if [ $LIST_ONLY -eq 0 ] && [ $DOWNLOAD_ONLY -eq 0 ]; then
        print_green "\nResults report is available at: ./reports/experiment_report.html"
        
        # If on macOS, automatically open the report
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open ./reports/experiment_report.html
        fi
    fi
else
    print_red "\nResults processing encountered errors."
    print_yellow "Check the output above for details."
fi 