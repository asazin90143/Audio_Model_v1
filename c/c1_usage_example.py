# Initialize
pipeline = EnvironmentalAudioPipeline(device='cuda')

# Process audio file (handles mono or multi-channel)
results = pipeline.process('input.wav', output_dir='./output')

# Access results
for event in results['events']:
    print(f"{event.label}: {event.start_time:.2f}s - {event.end_time:.2f}s")
    if event.azimuth:
        print(f"  Direction: {event.azimuth:.0f}° azimuth, {event.elevation:.0f}° elevation")