# log_health_scores.py - FIXED VERSION
import serial
import csv
import time
import re
import numpy as np

# Configure your serial port
SERIAL_PORT = 'COM3'  # Change if needed

def main():
    print("="*60)
    print("Health Score Logger for Q1 Journal")
    print("="*60)
    
    try:
        ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
        time.sleep(2)
        print(f"[OK] Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"[ERROR] Cannot open {SERIAL_PORT}: {e}")
        return
    
    health_scores = []
    anomaly_flags = []
    
    print("\n[INFO] Logging health scores for 60 seconds...")
    print("Press Ctrl+C to stop early\n")
    
    start_time = time.time()
    sample_count = 0
    
    try:
        while time.time() - start_time < 60:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                # Extract health score
                health_match = re.search(r'Health:\s*(\d+)%', line)
                anomaly_match = re.search(r'(NORMAL|ANOMALY)', line)
                
                if health_match:
                    health = int(health_match.group(1))
                    is_anomaly = 1 if anomaly_match and anomaly_match.group(1) == 'ANOMALY' else 0
                    health_scores.append(health)
                    anomaly_flags.append(is_anomaly)
                    sample_count += 1
                    
                    # Show progress every 10 samples
                    if sample_count % 10 == 0:
                        print(f"[{sample_count:4d}] Health: {health}%, Status: {anomaly_match.group(1) if anomaly_match else '?'}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Stopped early by user")
    
    ser.close()
    
    if len(health_scores) == 0:
        print("\n[ERROR] No health scores received!")
        print("Make sure:")
        print("1. ESP32 is connected and running")
        print("2. hil_player.py is running in another terminal")
        print("3. Both scripts are using the same COM port")
        return
    
    print(f"\n[SUMMARY] Logged {len(health_scores)} samples")
    
    # Separate normal and anomaly scores
    normal_scores = [h for h, a in zip(health_scores, anomaly_flags) if a == 0]
    anomaly_scores = [h for h, a in zip(health_scores, anomaly_flags) if a == 1]
    
    print(f"\nNormal operation samples: {len(normal_scores)}")
    if normal_scores:
        print(f"  Health score - Mean: {np.mean(normal_scores):.1f}%")
        print(f"               Std: {np.std(normal_scores):.1f}%")
        print(f"               Min: {np.min(normal_scores)}%")
        print(f"               Max: {np.max(normal_scores)}%")
    
    print(f"\nAnomaly samples: {len(anomaly_scores)}")
    if anomaly_scores:
        print(f"  Health score - Mean: {np.mean(anomaly_scores):.1f}%")
        print(f"               Std: {np.std(anomaly_scores):.1f}%")
        print(f"               Min: {np.min(anomaly_scores)}%")
        print(f"               Max: {np.max(anomaly_scores)}%")
    
    # Save to CSV
    with open('health_scores.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['health_score', 'is_anomaly'])
        writer.writerows(zip(health_scores, anomaly_flags))
    
    print("\n[OK] Saved to health_scores.csv")
    
    # Plot the results
    plot_health_distribution(health_scores, anomaly_scores, normal_scores)

def plot_health_distribution(health_scores, anomaly_scores, normal_scores):
    """Create health score visualizations"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Boxplot
    data_to_plot = [normal_scores, anomaly_scores]
    bp = axes[0].boxplot(data_to_plot, labels=['NORMAL', 'ANOMALY'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#90EE90')
    bp['boxes'][1].set_facecolor('#FFB6C1')
    axes[0].set_ylabel('Health Score (%)')
    axes[0].set_title('Health Score Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Histogram
    axes[1].hist(normal_scores, bins=20, alpha=0.5, label='NORMAL', color='green', edgecolor='black')
    axes[1].hist(anomaly_scores, bins=20, alpha=0.5, label='ANOMALY', color='red', edgecolor='black')
    axes[1].set_xlabel('Health Score (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Health Score Histogram')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Time series
    axes[2].plot(health_scores, 'b-', linewidth=1, alpha=0.7)
    axes[2].axhline(y=50, color='orange', linestyle='--', label='Warning (50%)')
    axes[2].axhline(y=30, color='red', linestyle='--', label='Critical (30%)')
    axes[2].set_xlabel('Sample Number')
    axes[2].set_ylabel('Health Score (%)')
    axes[2].set_title('Health Score Timeline')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('health_score_analysis.png', dpi=300)
    print("[OK] Saved health_score_analysis.png")
    
    # Also create violin plot separately
    plt.figure(figsize=(8, 6))
    data_for_violin = [normal_scores, anomaly_scores]
    parts = plt.violinplot(data_for_violin, positions=[0, 1], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('#90EE90')
    parts['bodies'][1].set_facecolor('#FFB6C1')
    plt.xticks([0, 1], ['NORMAL', 'ANOMALY'])
    plt.ylabel('Health Score (%)')
    plt.title('Health Score Distribution: Normal vs Anomaly')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('health_score_violin.png', dpi=300)
    print("[OK] Saved health_score_violin.png")

if __name__ == "__main__":
    main()