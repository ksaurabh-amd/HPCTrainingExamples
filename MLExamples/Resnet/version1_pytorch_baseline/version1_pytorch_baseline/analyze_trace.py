import json
import sys
from collections import defaultdict

class TraceAnalyzer:
    """
    Analyze PyTorch profiler trace files and group CUDA kernels by name, grid, and block dimensions.
    """
    
    def __init__(self, trace_file):
        """
        Initialize the TraceAnalyzer with a trace file.
        
        Args:
            trace_file (str): Path to the PyTorch profiler trace JSON file
        """
        self.trace_file = trace_file
        self.trace_data = None
        self.trace_events = None
        self.grouped_kernels = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'min_duration': float('inf'),
            'max_duration': 0,
            'events': []
        })
    
    def load_trace(self):
        """Load the trace file from disk."""
        try:
            with open(self.trace_file, "r") as f:
                self.trace_data = json.load(f)
            self.trace_events = self.trace_data.get("traceEvents", [])
            print(f"Loaded trace file: {self.trace_file}")
            print(f"Total trace events: {len(self.trace_events)}")
        except FileNotFoundError:
            print(f"Error: File '{self.trace_file}' not found.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: File '{self.trace_file}' is not a valid JSON file.")
            sys.exit(1)
    
    def analyze_kernels(self):
        """Analyze and group CUDA runtime kernel events."""
        if self.trace_events is None:
            print("Error: No trace data loaded. Call load_trace() first.")
            return
        
        for event in self.trace_events:
            # Filter for cuda_runtime category
            if event.get('cat') != 'cuda_runtime':
                continue
            
            args = event.get('args', {})
            
            # Check if kernel exists in args
            if 'kernel' not in args:
                continue
            
            # Extract kernel information
            kernel_name = args['kernel']
            grid = tuple(args.get('grid', []))
            block = tuple(args.get('block', []))
            duration = event.get('dur', 0)
            
            # Create grouping key
            group_key = (kernel_name, grid, block)
            
            # Update statistics
            self.grouped_kernels[group_key]['count'] += 1
            self.grouped_kernels[group_key]['total_duration'] += duration
            self.grouped_kernels[group_key]['min_duration'] = min(
                self.grouped_kernels[group_key]['min_duration'], duration
            )
            self.grouped_kernels[group_key]['max_duration'] = max(
                self.grouped_kernels[group_key]['max_duration'], duration
            )
            self.grouped_kernels[group_key]['events'].append(event)
    
    def _truncate_kernel_name(self, name, max_length=50):
        """Truncate long kernel names intelligently."""
        if len(name) <= max_length:
            return name
        
        # For long parametrized names, keep prefix and suffix
        # if '_' in name:
        #     parts = name.split('_')
        #     if len(parts) > 3:
        #         # Keep first 2 and last part
        #         return f"{parts[0]}_{parts[1]}_..._{parts[-1]}"[:max_length]
        
        # Default truncation
        return name[:max_length-3] + "..."
    
    def print_summary(self, max_rows=50, truncate_kernels=True):
        """
        Print a summary of grouped kernel events.
        
        Args:
            max_rows (int): Maximum number of rows to display
            truncate_kernels (bool): Whether to truncate long kernel names
        """
        if not self.grouped_kernels:
            print("No kernel events found in trace.")
            return
        
        print("\n" + "=" * 130)
        print(f"{'Kernel Name':<50} {'Grid':<18} {'Block':<18} {'Count':>8} {'Total(ms)':>12} {'Avg(us)':>10} {'%Time':>8}")
        print("=" * 130)
        
        # Sort by total duration (descending)
        sorted_kernels = sorted(
            self.grouped_kernels.items(),
            key=lambda x: x[1]['total_duration'],
            reverse=True
        )
        
        total_count = 0
        total_duration = 0
        
        # Calculate total duration for percentage
        grand_total_duration = sum(stats['total_duration'] for _, stats in sorted_kernels)
        
        for idx, ((kernel_name, grid, block), stats) in enumerate(sorted_kernels):
            if idx >= max_rows:
                remaining = len(sorted_kernels) - max_rows
                print(f"... {remaining} more kernel configurations not shown")
                break
            
            # Format kernel name
            if truncate_kernels:
                display_name = self._truncate_kernel_name(kernel_name, max_length=50)
            else:
                display_name = kernel_name[:50]
            
            # Format grid and block
            grid_str = f"[{','.join(map(str, grid))}]"
            block_str = f"[{','.join(map(str, block))}]"
            
            # Calculate statistics
            avg_duration = stats['total_duration'] / stats['count'] if stats['count'] > 0 else 0
            percentage = (stats['total_duration'] / grand_total_duration * 100) if grand_total_duration > 0 else 0
            
            print(f"{display_name:<50} {grid_str:<18} {block_str:<18} "
                  f"{stats['count']:>8} {stats['total_duration']/1000:>12.3f} "
                  f"{avg_duration:>10.3f} {percentage:>7.2f}%")
            
            total_count += stats['count']
            total_duration += stats['total_duration']
        
        print("=" * 130)
        print(f"Total: {len(self.grouped_kernels)} unique kernel configurations, "
              f"{total_count} total calls, {total_duration/1000:.3f} ms")
        print("=" * 130)
    
    def print_top_kernels(self, top_n=10):
        """Print only the top N most expensive kernels."""
        if not self.grouped_kernels:
            print("No kernel events found in trace.")
            return
        
        sorted_kernels = sorted(
            self.grouped_kernels.items(),
            key=lambda x: x[1]['total_duration'],
            reverse=True
        )[:top_n]
        
        grand_total = sum(stats['total_duration'] for _, stats in self.grouped_kernels.items())
        
        print(f"\n{'='*100}")
        print(f"Top {top_n} Most Expensive GPU Kernels")
        print(f"{'='*100}")
        print(f"{'Rank':<6} {'Kernel Name':<45} {'Total(ms)':>12} {'Calls':>8} {'Avg(us)':>10} {'%Time':>8}")
        print("-" * 100)
        
        for rank, ((kernel_name, grid, block), stats) in enumerate(sorted_kernels, 1):
            display_name = self._truncate_kernel_name(kernel_name, max_length=20)
            avg_duration = stats['total_duration'] / stats['count'] if stats['count'] > 0 else 0
            percentage = (stats['total_duration'] / grand_total * 100) if grand_total > 0 else 0
            
            print(f"{rank:<6} {display_name:<45} {stats['total_duration']/1000:>12.3f} "
                  f"{stats['count']:>8} {avg_duration:>10.3f} {percentage:>7.2f}%")
        
        print("=" * 100)
    
    def get_statistics(self):
        """
        Get statistics about the analyzed kernels.
        
        Returns:
            dict: Dictionary containing summary statistics
        """
        if not self.grouped_kernels:
            return {}
        
        total_count = sum(stats['count'] for stats in self.grouped_kernels.values())
        total_duration = sum(stats['total_duration'] for stats in self.grouped_kernels.values())
        
        return {
            'unique_kernels': len(self.grouped_kernels),
            'total_calls': total_count,
            'total_duration_ms': total_duration / 1000,
            'avg_duration_per_call_us': (total_duration / total_count) if total_count > 0 else 0
        }
    
    def run(self, top_n=20, show_full_table=False):
        """
        Run the complete analysis pipeline.
        
        Args:
            top_n (int): Number of top kernels to display
            show_full_table (bool): Whether to show the full kernel table
        """
        self.load_trace()
        self.analyze_kernels()
        
        # Show top N kernels (cleaner view)
        # self.print_top_kernels(top_n=top_n)
        
        # Optionally show full table
        if show_full_table:
            self.print_summary(max_rows=100, truncate_kernels=True)
        
        return self.get_statistics()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_trace.py <trace_file.json> [--full] [--top N]")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    show_full = '--full' in sys.argv
    
    # Parse top N argument
    top_n = 20
    if '--top' in sys.argv:
        try:
            top_idx = sys.argv.index('--top')
            top_n = int(sys.argv[top_idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --top argument, using default of 20")
    
    # Create analyzer and run analysis
    analyzer = TraceAnalyzer(trace_file)
    stats = analyzer.run(top_n=top_n, show_full_table=show_full)
    
    # Print additional statistics
    if stats:
        print(f"\n{'='*80}")
        print(f"Summary Statistics:")
        print(f"  Unique kernel configurations: {stats['unique_kernels']}")
        print(f"  Total kernel calls: {stats['total_calls']}")
        print(f"  Total GPU time: {stats['total_duration_ms']:.3f} ms")
        print(f"  Average duration per call: {stats['avg_duration_per_call_us']:.3f} us")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()