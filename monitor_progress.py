import time
import glob

target = 54
while True:
    count = len(glob.glob('results/phase1_non_reversal/*.pkl'))
    print(f"Progress: {count}/{target}")
    if count >= target:
        print("Complete!")
        break
    time.sleep(10)
