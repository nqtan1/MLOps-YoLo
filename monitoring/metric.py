from prometheus_client import Gauge 

cpu_usage = Gauge('cpu_usage', 'CPU Usage')
memory_usage = Gauge('memory_usage', 'Memory Usage')
disk_usage = Gauge('disk_usage', 'Disk Usage')
net_usage = Gauge('net_usage', 'Network Usage')


