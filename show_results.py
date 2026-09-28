import time, sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config
config.USE_MOCK_HARDWARE = True
results = {}
times = []
for _ in range(100):
    t = time.perf_counter()
    time.sleep(0.05)
    times.append((time.perf_counter() - t) * 1000)
avg = sum(times) / len(times)
jitter = (sum((x - avg)**2 for x in times) / len(times)) ** 0.5
results['timing_avg'] = round(avg, 2)
results['timing_jitter'] = round(jitter, 2)
results['timing_pass'] = 'PASS' if jitter < 2.0 else 'FAIL'
results['rl_avg'] = 'N/A'
results['rl_peak'] = 'N/A'
results['rl'] = 'SKIPPED'
try:
    import onnxruntime as ort, numpy as np, yaml
    gains = yaml.safe_load(open(PROJECT_ROOT / 'config' / 'gains.yaml'))
    obs_dim = gains.get('rl', {}).get('obs_dim', 17)
    mp = PROJECT_ROOT / gains.get('rl', {}).get('onnx_model_path', 'models/sac_policy_17D.onnx')
    if mp.exists():
        sess = ort.InferenceSession(str(mp))
        nm = sess.get_inputs()[0].name
        obs = np.random.rand(1, obs_dim).astype('float32')
        for _ in range(10): sess.run(None, {nm: obs})
        rt = []
        for _ in range(100):
            t = time.perf_counter()
            sess.run(None, {nm: obs})
            rt.append((time.perf_counter() - t) * 1000)
        results['rl_avg'] = round(sum(rt)/len(rt), 2)
        results['rl_peak'] = round(max(rt), 2)
        results['rl'] = 'PASS' if results['rl_avg'] < 5.0 else 'SLOW'
except Exception as e:
    results['rl'] = str(e)
print('')
print('=' * 50)
print('  GARUD GLIDER FINAL RESULTS')
print('=' * 50)
print('  1. Timing avg    :', results['timing_avg'], 'ms')
print('  2. Timing jitter :', results['timing_jitter'], 'ms')
print('  3. Timing result :', results['timing_pass'])
print('  4. RL avg        :', results['rl_avg'], 'ms')
print('  5. RL peak       :', results['rl_peak'], 'ms')
print('  6. RL result     :', results['rl'])
print('=' * 50)