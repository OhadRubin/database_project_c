#!/bin/bash

dirs=($(sudo du -h --max-depth=1 ~/ | sort -h | awk '{print $2}' | tail -n 20 | head -n 19))
# dirs=($(sudo du -h --max-depth=1 /home/ohadr/.local | sort -h | awk '{print $2}' | tail -n 10 | head -n 9))
for dir in "${dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Warning: $dir is not a directory or doesnt exist"
        continue
    fi
    
    echo "Largest file in $dir:"
    total_size=$(du -sh "$dir" | awk '{print $1}')
    echo "Total size: $total_size"
    find "$dir" -type f -printf "%s %p\n" 2>/dev/null | sort -nr | head -n1 | awk '{printf "%.2f MB (%d bytes): %s\n", $1/1024/1024, $1, substr($0, length($1)+2)}'
    echo ""
done 