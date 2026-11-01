s = '2026-11-01T13:45:09'
date, time = s.split('T')
y, m, d = map(int, date.split('-'))
hh, mm, ss = [int(p) for p in time.split(':')]
secs = hh * 3600 + mm * 60 + ss
print(y, m, d, secs, f'{d:02d}/{m:02d}/{y}', f'{secs // 3600}h{secs % 3600 // 60:02d}m', divmod(secs, 60), f'{hh % 12 or 12}:{mm:02d} {"AM" if hh < 12 else "PM"}')
