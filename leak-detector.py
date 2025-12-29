#!/usr/bin/env python3
"""
Memory Leak Detection Agent
A lightweight daemon that monitors processes for memory leaks using statistical analysis.
"""

import psutil
import time
import logging
import argparse
import signal
import sys
import json
from datetime import datetime
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import os


@dataclass
class MemorySnapshot:
    """Single memory measurement snapshot"""
    timestamp: float
    rss_mb: float
    vms_mb: float
    percent: float


@dataclass
class ProcessStats:
    """Statistical analysis of process memory"""
    pid: int
    name: str
    current_rss_mb: float
    moving_avg_rss_mb: float
    std_dev: float
    growth_rate: float  # MB per minute
    anomaly_score: float
    samples_count: int


class MemoryLeakDetector:
    """Detects memory leaks using statistical pattern analysis"""
    
    def __init__(self, 
                 window_size: int = 30,
                 anomaly_threshold: float = 2.5,
                 min_samples: int = 10,
                 growth_threshold: float = 0.5):
        """
        Args:
            window_size: Number of samples to keep in moving window
            anomaly_threshold: Standard deviations for anomaly detection
            min_samples: Minimum samples before detecting anomalies
            growth_threshold: Minimum MB/min growth to consider anomalous
        """
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.min_samples = min_samples
        self.growth_threshold = growth_threshold
        self.process_history: Dict[int, deque] = {}
        
    def add_sample(self, pid: int, snapshot: MemorySnapshot):
        """Add a memory snapshot for a process"""
        if pid not in self.process_history:
            self.process_history[pid] = deque(maxlen=self.window_size)
        self.process_history[pid].append(snapshot)
    
    def _calculate_stats(self, samples: deque) -> tuple:
        """Calculate moving average and standard deviation"""
        if len(samples) < 2:
            return 0.0, 0.0
        
        rss_values = [s.rss_mb for s in samples]
        mean = sum(rss_values) / len(rss_values)
        variance = sum((x - mean) ** 2 for x in rss_values) / len(rss_values)
        std_dev = variance ** 0.5
        
        return mean, std_dev
    
    def _calculate_growth_rate(self, samples: deque) -> float:
        """Calculate memory growth rate in MB per minute"""
        if len(samples) < 2:
            return 0.0
        
        # Linear regression for growth rate
        first = samples[0]
        last = samples[-1]
        time_diff = (last.timestamp - first.timestamp) / 60  # minutes
        
        if time_diff == 0:
            return 0.0
        
        mem_diff = last.rss_mb - first.rss_mb
        return mem_diff / time_diff
    
    def analyze(self, pid: int, process_name: str) -> Optional[ProcessStats]:
        """Analyze process memory and detect anomalies"""
        if pid not in self.process_history:
            return None
        
        samples = self.process_history[pid]
        if len(samples) < self.min_samples:
            return None
        
        current = samples[-1]
        mean, std_dev = self._calculate_stats(samples)
        growth_rate = self._calculate_growth_rate(samples)
        
        # Calculate anomaly score (how many std devs from mean)
        anomaly_score = 0.0
        if std_dev > 0:
            anomaly_score = abs(current.rss_mb - mean) / std_dev
        
        return ProcessStats(
            pid=pid,
            name=process_name,
            current_rss_mb=current.rss_mb,
            moving_avg_rss_mb=mean,
            std_dev=std_dev,
            growth_rate=growth_rate,
            anomaly_score=anomaly_score,
            samples_count=len(samples)
        )
    
    def is_leak_detected(self, stats: ProcessStats) -> tuple[bool, str]:
        """Determine if memory leak is detected and return reason"""
        reasons = []
        
        # Check for statistical anomaly
        if stats.anomaly_score > self.anomaly_threshold:
            reasons.append(f"Anomaly score {stats.anomaly_score:.2f}σ exceeds threshold")
        
        # Check for sustained growth
        if stats.growth_rate > self.growth_threshold:
            reasons.append(f"Growth rate {stats.growth_rate:.2f} MB/min exceeds threshold")
        
        # Combined detection: both anomaly and growth
        is_leak = len(reasons) >= 2 or (
            stats.anomaly_score > self.anomaly_threshold * 1.5
        ) or (
            stats.growth_rate > self.growth_threshold * 2
        )
        
        return is_leak, "; ".join(reasons) if reasons else "Normal"


class MemoryMonitorDaemon:
    """Main daemon that monitors processes"""
    
    def __init__(self, 
                 interval: int = 10,
                 target_pids: List[int] = None,
                 target_names: List[str] = None,
                 log_file: str = "memory_monitor.log",
                 alert_log: str = "memory_alerts.json"):
        """
        Args:
            interval: Seconds between checks
            target_pids: Specific PIDs to monitor (None = monitor all)
            target_names: Process names to monitor (None = monitor all)
            log_file: Path to main log file
            alert_log: Path to alert log file (JSON)
        """
        self.interval = interval
        self.target_pids = set(target_pids) if target_pids else None
        self.target_names = set(target_names) if target_names else None
        self.log_file = log_file
        self.alert_log = alert_log
        self.detector = MemoryLeakDetector()
        self.running = False
        self.alerted_pids: Dict[int, float] = {}  # Track last alert time
        
        self._setup_logging()
        self._setup_signal_handlers()
    
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def _should_monitor(self, proc: psutil.Process) -> bool:
        """Check if process should be monitored"""
        try:
            if self.target_pids and proc.pid not in self.target_pids:
                return False
            if self.target_names and proc.name() not in self.target_names:
                return False
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def _get_memory_snapshot(self, proc: psutil.Process) -> Optional[MemorySnapshot]:
        """Get memory snapshot for a process"""
        try:
            mem_info = proc.memory_info()
            mem_percent = proc.memory_percent()
            
            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=mem_info.rss / (1024 * 1024),
                vms_mb=mem_info.vms / (1024 * 1024),
                percent=mem_percent
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def _log_alert(self, stats: ProcessStats, reason: str):
        """Log memory leak alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "pid": stats.pid,
            "process_name": stats.name,
            "current_memory_mb": stats.current_rss_mb,
            "average_memory_mb": stats.moving_avg_rss_mb,
            "growth_rate_mb_per_min": stats.growth_rate,
            "anomaly_score": stats.anomaly_score,
            "reason": reason
        }
        
        # Log to console
        self.logger.warning(
            f"⚠️  MEMORY LEAK DETECTED - PID: {stats.pid} ({stats.name}) | "
            f"Current: {stats.current_rss_mb:.1f} MB | "
            f"Avg: {stats.moving_avg_rss_mb:.1f} MB | "
            f"Growth: {stats.growth_rate:.2f} MB/min | "
            f"Reason: {reason}"
        )
        
        # Append to JSON log
        try:
            with open(self.alert_log, 'a') as f:
                json.dump(alert, f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to write alert log: {e}")
    
    def _should_alert(self, pid: int) -> bool:
        """Check if we should alert (avoid spam)"""
        current_time = time.time()
        last_alert = self.alerted_pids.get(pid, 0)
        
        # Alert at most once per 5 minutes per process
        if current_time - last_alert < 300:
            return False
        
        self.alerted_pids[pid] = current_time
        return True
    
    def monitor_cycle(self):
        """Single monitoring cycle"""
        monitored_count = 0
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if not self._should_monitor(proc):
                    continue
                
                snapshot = self._get_memory_snapshot(proc)
                if not snapshot:
                    continue
                
                pid = proc.pid
                name = proc.name()
                
                # Add sample and analyze
                self.detector.add_sample(pid, snapshot)
                stats = self.detector.analyze(pid, name)
                
                if stats:
                    is_leak, reason = self.detector.is_leak_detected(stats)
                    
                    if is_leak and self._should_alert(pid):
                        self._log_alert(stats, reason)
                    elif monitored_count < 5:  # Log first few for visibility
                        self.logger.debug(
                            f"Monitoring PID: {pid} ({name}) | "
                            f"RSS: {stats.current_rss_mb:.1f} MB | "
                            f"Growth: {stats.growth_rate:.2f} MB/min"
                        )
                
                monitored_count += 1
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if monitored_count > 0:
            self.logger.info(f"Monitoring cycle complete: {monitored_count} processes checked")
    
    def run(self):
        """Start the monitoring daemon"""
        self.running = True
        self.logger.info("Memory Leak Detection Agent started")
        self.logger.info(f"Monitoring interval: {self.interval}s")
        self.logger.info(f"Alert log: {self.alert_log}")
        
        if self.target_pids:
            self.logger.info(f"Monitoring specific PIDs: {self.target_pids}")
        if self.target_names:
            self.logger.info(f"Monitoring process names: {self.target_names}")
        
        try:
            while self.running:
                self.monitor_cycle()
                time.sleep(self.interval)
        except Exception as e:
            self.logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            self.logger.info("Memory Leak Detection Agent stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Memory Leak Detection Agent - Monitor processes for memory leaks"
    )
    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=10,
        help='Monitoring interval in seconds (default: 10)'
    )
    parser.add_argument(
        '-p', '--pids',
        type=int,
        nargs='+',
        help='Specific process IDs to monitor'
    )
    parser.add_argument(
        '-n', '--names',
        nargs='+',
        help='Process names to monitor (e.g., python, node)'
    )
    parser.add_argument(
        '-l', '--log-file',
        default='memory_monitor.log',
        help='Path to log file (default: memory_monitor.log)'
    )
    parser.add_argument(
        '-a', '--alert-log',
        default='memory_alerts.json',
        help='Path to alert log file (default: memory_alerts.json)'
    )
    
    args = parser.parse_args()
    
    daemon = MemoryMonitorDaemon(
        interval=args.interval,
        target_pids=args.pids,
        target_names=args.names,
        log_file=args.log_file,
        alert_log=args.alert_log
    )
    
    daemon.run()


if __name__ == '__main__':
    main()
