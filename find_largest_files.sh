#!/bin/bash

dirs=($(sudo du -h --max-depth=1 ~/ | sort -h | awk '{print $2}' | tail -n 10 | head -n 9))
for dir in "${dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Warning: $dir is not a directory or doesnt exist"
        continue
    fi
    
    echo "Largest file in $dir:"
    find "$dir" -type f -printf "%s %p\n" 2>/dev/null | sort -nr | head -n1 | awk '{printf "%.2f MB (%d bytes): %s\n", $1/1024/1024, $1, substr($0, length($1)+2)}'
    echo ""
done 