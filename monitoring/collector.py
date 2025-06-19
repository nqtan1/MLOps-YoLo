import time 
from prometheus_client import start_http_server
import metric
import psutil  # Get system information

def collect_metrics():
    # Start the Prometheus HTTP server
    start_http_server(8000)  # Prometheus will scrape metrics from this port
    print("-> Prometheus metrics server started on port 8000.")
    
    while True:
        # Update CPU usage
        metric.cpu_usage.set(psutil.cpu_percent(interval=1))
        
        # Update memory usage
        metric.memory_usage.set(psutil.virtual_memory().percent)
        
        # Update disk usage
        metric.disk_usage.set(psutil.disk_usage('/').percent)
        
        # Update network usage (example for eth0 interface)
        
        time.sleep(5)  # Sleep for a while before collecting metrics again

if __name__ == "__main__":
    collect_metrics()