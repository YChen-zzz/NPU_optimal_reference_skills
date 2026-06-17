# Profiling 分析脚本模板

## kernel_details 分析

```python
import csv
from collections import defaultdict

kernels = []
with open('kernel_details.csv') as f:
    for r in csv.DictReader(f): kernels.append(r)

target_step = 2  # 选稳态 step
step_n = [r for r in kernels if int(r['Step Id']) == target_step]
total_wait = sum(float(r['Wait Time(us)']) for r in step_n)
total_dur = sum(float(r['Duration(us)']) for r in step_n)
print(f"kernels: {len(step_n)}, stall_ratio: {total_wait/total_dur*100:.0f}%")
print(f"avg wait/kernel: {total_wait/len(step_n):.0f}us")

# wait time 分布
for lo, hi in [(0,10),(10,50),(50,100),(100,200),(200,500),(500,9999)]:
    cnt = sum(1 for r in step_n if lo <= float(r['Wait Time(us)']) < hi)
    tw = sum(float(r['Wait Time(us)']) for r in step_n if lo <= float(r['Wait Time(us)']) < hi)
    print(f"  {lo}-{hi}us: {cnt} kernels, wait={tw/1000:.1f}ms")
```

## operator_details host-only op 分析

```python
ops = []
with open('operator_details.csv') as f:
    for row in csv.DictReader(f): ops.append(row)

# 分离有 device kernel 的 op 和纯 host op
kernel_ops = [o for o in ops if float(o.get('Device Self Duration(us)', 0) or 0) > 0.1]
host_only = [o for o in ops if float(o.get('Device Self Duration(us)', 0) or 0) <= 0.1]

total_hostonly = sum(float(o.get('Host Self Duration(us)', 0) or 0) for o in host_only)

by_type = defaultdict(lambda: {'count': 0, 'host': 0})
for o in host_only:
    by_type[o['Name']]['count'] += 1
    by_type[o['Name']]['host'] += float(o.get('Host Self Duration(us)', 0) or 0)

n_steps = 3  # 按实际 profiling steps 调整
print(f"{'Op':40s} {'Count/step':>10} {'Host/step(ms)':>12} {'Avg(us)':>8}")
for name, s in sorted(by_type.items(), key=lambda x: -x[1]['host'])[:20]:
    print(f"  {name:38s} {s['count']//n_steps:>10} {s['host']/n_steps/1000:>12.2f} {s['host']/max(s['count'],1):>8.1f}")
```

## 按算子类型聚合 bubble 分布

```python
# 从 kernel_details 按算子名称聚合 wait time
by_name = defaultdict(lambda: {'count': 0, 'wait': 0, 'dur': 0})
for r in step_n:
    name = r['Name']
    by_name[name]['count'] += 1
    by_name[name]['wait'] += float(r['Wait Time(us)'])
    by_name[name]['dur'] += float(r['Duration(us)'])

grand_total = sum(v['wait'] for v in by_name.values())
print(f"{'Kernel Type':40s} {'Count':>6} {'Wait(ms)':>10} {'Pct':>6}")
for name, s in sorted(by_name.items(), key=lambda x: -x[1]['wait'])[:15]:
    print(f"  {name:38s} {s['count']:>6} {s['wait']/1000:>10.1f} {s['wait']/grand_total*100:>5.1f}%")
```

## 跨版本对比分析

```python
def load_operator_stats(csv_path, n_steps):
    stats = defaultdict(lambda: {'count': 0, 'host_self': 0})
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            name = row['Name']
            stats[name]['count'] += 1
            stats[name]['host_self'] += float(row.get('Host Self Duration(us)', 0) or 0)
    # 归一化到每步
    for v in stats.values():
        v['count'] //= n_steps
        v['host_self'] /= n_steps
    return dict(stats)

old = load_operator_stats('old/operator_details.csv', 3)
new = load_operator_stats('new/operator_details.csv', 3)
all_ops = set(old) | set(new)

print(f"{'Op':35s} {'Old(ms)':>8} {'New(ms)':>8} {'Delta':>8}")
deltas = []
for op in all_ops:
    o = old.get(op, {'host_self': 0})['host_self'] / 1000
    n = new.get(op, {'host_self': 0})['host_self'] / 1000
    deltas.append((op, o, n, n - o))

for op, o, n, d in sorted(deltas, key=lambda x: x[3]):
    if abs(d) > 0.1:
        print(f"  {op:33s} {o:>8.2f} {n:>8.2f} {d:>+8.2f}")
```
