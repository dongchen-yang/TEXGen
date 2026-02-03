#!/usr/bin/env python3
"""
Analyze memory tracking logs to identify memory leaks.

Usage:
    python analyze_memory_logs.py logs/lightgen_12345.out
    
Or to monitor in real-time:
    tail -f logs/lightgen_12345.out | python analyze_memory_logs.py --realtime
"""

import argparse
import re
import sys
from collections import defaultdict
from typing import List, Tuple


def parse_memory_log(line: str) -> dict:
    """Parse a memory log line into structured data."""
    # Example: [MEMORY] train_step_start (epoch=0, batch=0) | Allocated: 18.45 GB (Δ +3.22) | Reserved: 19.00 GB | Free: 60.73 GB | Peak: 18.45 GB
    
    pattern = r'\[MEMORY\]\s+(.+?)\s+\|\s+Allocated:\s+([\d.]+)\s+GB(?:\s+\(Δ\s*([+-]?[\d.]+)\))?\s+\|\s+Reserved:\s+([\d.]+)\s+GB.*?\|\s+Free:\s+([\d.]+)\s+GB\s+\|\s+Peak:\s+([\d.]+)\s+GB'
    
    match = re.search(pattern, line)
    if not match:
        return None
    
    tag, allocated, delta, reserved, free, peak = match.groups()
    
    return {
        'tag': tag.strip(),
        'allocated': float(allocated),
        'delta': float(delta) if delta else None,
        'reserved': float(reserved),
        'free': float(free),
        'peak': float(peak),
    }


def extract_epoch_batch(tag: str) -> Tuple[int, int]:
    """Extract epoch and batch number from tag."""
    epoch_match = re.search(r'epoch=(\d+)', tag)
    batch_match = re.search(r'batch=(\d+)', tag)
    
    epoch = int(epoch_match.group(1)) if epoch_match else None
    batch = int(batch_match.group(1)) if batch_match else None
    
    return epoch, batch


def analyze_logs(logfile: str):
    """Analyze memory logs and report findings."""
    
    # Storage for tracking
    epoch_memories = defaultdict(list)  # epoch -> list of allocated memories
    step_memories = []  # All step memories in order
    validation_memories = []
    
    # Parse log file
    with open(logfile, 'r') as f:
        for line in f:
            if '[MEMORY]' not in line:
                continue
            
            data = parse_memory_log(line)
            if not data:
                continue
            
            epoch, batch = extract_epoch_batch(data['tag'])
            
            # Track by epoch (for leak detection)
            if epoch is not None and 'train_step_end' in data['tag'] and batch == 0:
                epoch_memories[epoch].append(data['allocated'])
            
            # Track validation
            if 'validation' in data['tag'].lower():
                validation_memories.append(data)
            
            # Track all training steps
            if 'train_step' in data['tag']:
                step_memories.append(data)
    
    print("=" * 80)
    print("MEMORY LEAK ANALYSIS")
    print("=" * 80)
    
    # 1. Check for memory growth across epochs
    print("\n1. MEMORY GROWTH ACROSS EPOCHS")
    print("-" * 80)
    
    if len(epoch_memories) < 2:
        print("   Not enough epoch data to analyze trends (need at least 2 epochs)")
    else:
        epochs = sorted(epoch_memories.keys())
        
        # Sample every 5 epochs
        sample_epochs = [e for e in epochs if e % 5 == 0][:20]
        
        print(f"   Epoch    | Allocated (GB) | Change from Previous")
        print(f"   " + "-" * 60)
        
        prev_alloc = None
        for epoch in sample_epochs:
            if epoch_memories[epoch]:
                alloc = epoch_memories[epoch][0]
                if prev_alloc is not None:
                    change = alloc - prev_alloc
                    status = "⚠️ LEAK" if change > 0.5 else "✓ OK"
                    print(f"   {epoch:5d}    | {alloc:14.2f} | {change:+8.2f} GB  {status}")
                else:
                    print(f"   {epoch:5d}    | {alloc:14.2f} | (baseline)")
                prev_alloc = alloc
        
        # Calculate overall trend
        if len(sample_epochs) >= 2:
            first_epoch = sample_epochs[0]
            last_epoch = sample_epochs[-1]
            first_mem = epoch_memories[first_epoch][0]
            last_mem = epoch_memories[last_epoch][0]
            total_growth = last_mem - first_mem
            epochs_span = last_epoch - first_epoch
            
            if epochs_span > 0:
                growth_per_epoch = total_growth / epochs_span
                
                print(f"\n   Total growth: {total_growth:+.2f} GB over {epochs_span} epochs")
                print(f"   Average growth: {growth_per_epoch:+.3f} GB per epoch")
                
                if growth_per_epoch > 0.1:
                    print(f"   ⚠️  SIGNIFICANT LEAK DETECTED: {growth_per_epoch:.3f} GB/epoch")
                    epochs_to_oom = (79 - last_mem) / growth_per_epoch
                    print(f"   ⚠️  Estimated epochs until OOM: ~{epochs_to_oom:.0f} epochs")
                elif growth_per_epoch > 0.01:
                    print(f"   ⚠️  Small leak detected: {growth_per_epoch:.3f} GB/epoch")
                else:
                    print(f"   ✓ No significant leak detected")
    
    # 2. Check cleanup effectiveness
    print("\n2. CLEANUP EFFECTIVENESS")
    print("-" * 80)
    
    cleanup_pairs = []
    i = 0
    while i < len(step_memories) - 1:
        if 'before_batch_cleanup' in step_memories[i]['tag']:
            if 'after_batch_cleanup' in step_memories[i+1]['tag']:
                before = step_memories[i]['allocated']
                after = step_memories[i+1]['allocated']
                cleanup_pairs.append((before, after, after - before))
                i += 2
                continue
        i += 1
    
    if cleanup_pairs:
        avg_cleanup = sum(delta for _, _, delta in cleanup_pairs) / len(cleanup_pairs)
        max_before = max(before for before, _, _ in cleanup_pairs)
        min_after = min(after for _, after, _ in cleanup_pairs)
        
        print(f"   Cleanup operations analyzed: {len(cleanup_pairs)}")
        print(f"   Average memory freed: {-avg_cleanup:.3f} GB")
        print(f"   Max memory before cleanup: {max_before:.2f} GB")
        print(f"   Min memory after cleanup: {min_after:.2f} GB")
        
        if avg_cleanup > -0.1:
            print(f"   ⚠️  Cleanup not effective: only freeing {-avg_cleanup:.3f} GB on average")
        else:
            print(f"   ✓ Cleanup working: freeing {-avg_cleanup:.3f} GB per cleanup")
    else:
        print("   No cleanup pairs found in logs")
    
    # 3. Check peak memory
    print("\n3. PEAK MEMORY USAGE")
    print("-" * 80)
    
    if step_memories:
        peak_allocated = max(m['allocated'] for m in step_memories)
        peak_reserved = max(m['reserved'] for m in step_memories)
        min_free = min(m['free'] for m in step_memories)
        
        print(f"   Peak allocated: {peak_allocated:.2f} GB")
        print(f"   Peak reserved:  {peak_reserved:.2f} GB")
        print(f"   Min free:       {min_free:.2f} GB")
        print(f"   Fragmentation:  {peak_reserved - peak_allocated:.2f} GB")
        
        total_gpu = peak_allocated + min_free
        utilization = (peak_allocated / total_gpu) * 100
        
        print(f"   Total GPU:      {total_gpu:.2f} GB")
        print(f"   Utilization:    {utilization:.1f}%")
        
        if min_free < 2.0:
            print(f"   ⚠️  CRITICAL: Less than 2 GB free at peak")
        elif min_free < 5.0:
            print(f"   ⚠️  WARNING: Less than 5 GB free at peak")
        else:
            print(f"   ✓ Sufficient free memory")
        
        if peak_reserved - peak_allocated > 5.0:
            print(f"   ⚠️  High fragmentation: {peak_reserved - peak_allocated:.2f} GB wasted")
            print(f"   → Recommendation: Use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    # 4. Check validation memory
    print("\n4. VALIDATION MEMORY")
    print("-" * 80)
    
    if validation_memories:
        val_start = [m for m in validation_memories if 'validation_start' in m['tag']]
        val_end = [m for m in validation_memories if 'after_val_cleanup' in m['tag']]
        
        if val_start and val_end:
            avg_val_start = sum(m['allocated'] for m in val_start) / len(val_start)
            avg_val_end = sum(m['allocated'] for m in val_end) / len(val_end)
            
            print(f"   Validation starts: {len(val_start)}")
            print(f"   Avg memory at validation start: {avg_val_start:.2f} GB")
            print(f"   Avg memory after validation:    {avg_val_end:.2f} GB")
            print(f"   Memory retained per validation: {avg_val_end - avg_val_start:+.2f} GB")
            
            if avg_val_end - avg_val_start > 1.0:
                print(f"   ⚠️  Validation leaking {avg_val_end - avg_val_start:.2f} GB per run")
            else:
                print(f"   ✓ Validation cleanup working")
    else:
        print("   No validation logs found")
    
    # 5. Summary and recommendations
    print("\n5. SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    if len(epoch_memories) >= 2:
        epochs = sorted(epoch_memories.keys())
        if len(epochs) >= 2:
            first_mem = epoch_memories[epochs[0]][0]
            last_mem = epoch_memories[epochs[-1]][0]
            growth_rate = (last_mem - first_mem) / len(epochs)
            
            if growth_rate > 0.1:
                print("⚠️  CRITICAL LEAK DETECTED")
                print(f"   - Memory growing at {growth_rate:.3f} GB/epoch")
                print(f"   - Will OOM in ~{(79 - last_mem) / growth_rate:.0f} epochs")
                print("\n   Recommendations:")
                print("   1. Check which operation shows largest delta in logs")
                print("   2. Verify tensors are being deleted (check for 'del' statements)")
                print("   3. Ensure gc.collect() is being called")
                print("   4. Consider reducing batch size further")
            elif growth_rate > 0.01:
                print("⚠️  Small leak detected")
                print(f"   - Memory growing at {growth_rate:.3f} GB/epoch")
                print("   - Monitor for a few more epochs")
            else:
                print("✓ No significant leak detected!")
                print(f"   - Memory stable at {last_mem:.2f} GB")
                
                if peak_allocated > 70:
                    print("\n   However, peak memory is very high:")
                    print(f"   - Peak: {peak_allocated:.2f} GB (using {utilization:.0f}% of GPU)")
                    print("   - Consider reducing batch size for safety margin")


def analyze_realtime():
    """Analyze memory logs from stdin (for real-time monitoring)."""
    print("=" * 80)
    print("REAL-TIME MEMORY MONITORING")
    print("=" * 80)
    print("Watching for memory patterns...")
    print("")
    
    last_allocated = None
    leak_warnings = 0
    
    for line in sys.stdin:
        if '[MEMORY]' not in line:
            continue
        
        data = parse_memory_log(line)
        if not data:
            print(line.strip())
            continue
        
        # Colorize output based on memory usage
        allocated = data['allocated']
        free = data['free']
        
        status = ""
        if free < 2.0:
            status = " 🔴 CRITICAL"
        elif free < 5.0:
            status = " 🟠 WARNING"
        elif free < 10.0:
            status = " 🟡 LOW"
        else:
            status = " 🟢 OK"
        
        # Check for leak
        if last_allocated is not None and 'epoch_start' in data['tag']:
            growth = allocated - last_allocated
            if growth > 0.5:
                leak_warnings += 1
                print(f"⚠️  LEAK ALERT: Memory grew by {growth:.2f} GB since last epoch!")
        
        # Print the log line with status
        print(f"{data['tag']:45s} | {allocated:6.2f} GB | Free: {free:6.2f} GB {status}")
        
        # Track for next iteration
        if 'epoch_start' in data['tag']:
            last_allocated = allocated
        
        # Alert if too many leaks
        if leak_warnings >= 3:
            print("\n" + "="*80)
            print("⚠️  MULTIPLE LEAKS DETECTED - TRAINING LIKELY TO OOM")
            print("="*80)
            print("Recommendation: Stop training and investigate")
            print("")
            leak_warnings = 0  # Reset counter


def main():
    parser = argparse.ArgumentParser(description="Analyze memory tracking logs")
    parser.add_argument('logfile', nargs='?', help='Path to log file (omit for stdin)')
    parser.add_argument('--realtime', action='store_true', help='Real-time monitoring mode')
    
    args = parser.parse_args()
    
    if args.realtime or args.logfile is None:
        try:
            analyze_realtime()
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        analyze_logs(args.logfile)


if __name__ == "__main__":
    main()
