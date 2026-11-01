rows = [('apple', 3, 1.5), ('kiwi', 12, 0.25), ('watermelon', 1, 12.0)]
print(f"{'item':<12}{'qty':>5}{'price':>8}{'total':>9}")
print('-' * 34)
grand = 0
for name, qty, price in rows:
    total = qty * price
    grand += total
    print(f'{name:<12}{qty:>5d}{price:>8.2f}{total:>9.2f}')
print(f"{'':<12}{sum(r[1] for r in rows):>5}{'':>8}{grand:>9.2f}")
print('=' * 34, f'{grand / len(rows):.3f}', f'{max(rows, key=lambda r: r[1] * r[2])[0].upper()!r}')
