import json
import sys

# Get timestamp from command line or use default
timestamp = sys.argv[1] if len(sys.argv) > 1 else "20250529_125448"

# Load the intermediate results
try:
    with open(f'results/{timestamp}/intermediate/intermediate_results.json', 'r') as f:
        results = json.load(f)
except FileNotFoundError:
    print(f"No intermediate results found for timestamp {timestamp}")
    sys.exit(1)

# Filter out errored results (num_total = 0)
filtered_results = {}
removed_count = 0
for key, value in results.items():
    if value.get('num_total', 1) > 0:  # Keep only successful evaluations
        filtered_results[key] = value
    else:
        removed_count += 1
        print(f'Removing errored result: {key}')

# Save the filtered results back
with open(f'results/{timestamp}/intermediate/intermediate_results.json', 'w') as f:
    json.dump(filtered_results, f, indent=2)

print(f'\nRemoved {removed_count} errored results')
print(f'Kept {len(filtered_results)} successful results') 