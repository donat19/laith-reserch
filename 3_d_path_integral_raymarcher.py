"""
3D Path-Integral Raymarcher (CPU, pure Python + numpy)
----------------------------------------------------
Однофайловая реализация «рендера по пути» для демонстрации идеи:
- для каждого пикселя генерируем много случайных путей, проходящих от камеры в направлении пикселя;
- путь представлен как ломаная из сегментов с небольшим случайным сдвигом (моделирует все возможные траектории);
- каждый сегмент "маршрутизируется" через сцену с помощью дискретного sphere-tracing (SDF);
- если путь пересекает геометрию — считаем оптическую длину до точки пересечения и фазу exp(i k L);
- суммируем комплексные амплитуды по путям -> получаем интенсивность (|sum|^2) для пикселя.

Примечание: это учебный демонстратор — он НЕ ускорён под GPU и может работать медленно.
Для быстрого исследования уменьшите разрешение и количество путей на пиксель (spp).

Требования: Python 3.8+, numpy, Pillow
Запуск: python3 3D_path_integral_raymarcher.py

"""

import math
import sys
import numpy as np
from PIL import Image
from time import time

# ----------------------------- Сцена (SDF) -----------------------------

def sdf_sphere(p, center, r):
    return np.linalg.norm(p - center, axis=-1) - r


def sdf_plane(p, n, d):
    # plane: n·x + d = 0
    return (p @ n) + d


def sdf_box(p, center, size):
    q = np.abs(p - center) - size
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.maximum(q[:,0], np.maximum(q[:,1], q[:,2])), 0.0)


def scene_sdf(p):
    """
    Возвращает минимальное расстояние до геометрии и id материала.
    p: (...,3) numpy array
    """
    # Сфера 1
    d1 = sdf_sphere(p, np.array([0.0, 0.5, 4.0]), 0.8)
    # Сфера 2
    d2 = sdf_sphere(p, np.array([1.3, -0.2, 5.0]), 0.6)
    # Плоскость (пол)
    d3 = sdf_plane(p, np.array([0.0, 1.0, 0.0]), 0.8)

    # комбинируем (min)
    d = np.minimum(np.minimum(d1, d2), d3)

    # material id for shading (simple)
    mat = np.full(p.shape[:-1], 0, dtype=np.int32)
    mat[d2 == d] = 2
    mat[d1 == d] = 1
    mat[d3 == d] = 3

    return d, mat

# ----------------------------- Утилиты -----------------------------

def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def orthonormal_basis(forward):
    # возвращает два вектора, ортогональных к forward
    f = normalize(forward)
    if abs(f[0]) < 0.9:
        up = np.array([1.0, 0.0, 0.0])
    else:
        up = np.array([0.0, 1.0, 0.0])
    right = normalize(np.cross(up, f))
    up2 = np.cross(f, right)
    return right, up2

# ----------------------------- Рендерер -----------------------------

def render(width=200, height=120, fov=45.0, spp=40, segments=6, wavelength=0.5,
           jitter_scale=0.6, max_steps_per_segment=40, hit_eps=0.02, max_distance=50.0):
    start = time()
    aspect = width / height
    cam_pos = np.array([0.0, 0.0, 0.0])
    cam_dir = np.array([0.0, 0.0, 1.0])
    right, up = orthonormal_basis(cam_dir)

    # precompute k
    k = 2.0 * math.pi / wavelength

    img = np.zeros((height, width), dtype=np.float64)

    # координаты пикселей в экранном пространстве
    screen_h = 2.0 * math.tan(math.radians(fov) / 2.0)
    screen_w = screen_h * aspect

    total_pixels = width * height
    pix_idx = 0

    for y in range(height):
        for x in range(width):
            # Нормализованные координаты на экране [-0.5,0.5]
            u = (x + 0.5) / width - 0.5
            v = (y + 0.5) / height - 0.5
            # позиция на плоскости проекции
            pixel_center = cam_pos + cam_dir + right * (u * screen_w) + up * (v * screen_h)
            primary_dir = normalize(pixel_center - cam_pos)

            pixel_amp = 0 + 0j
            hits = 0

            for s in range(spp):
                # генерируем ломаную: N точек от камеры до далеко (primary ray * t_end)
                t_end = 30.0  # максимальная длинна пути
                t_vals = np.linspace(0.0, t_end, segments + 1)

                # базовые точки по прямой
                pts = np.outer(t_vals, primary_dir) + cam_pos

                # добавляем случайный шум к внутренним точкам (кроме концов)
                noise = np.zeros_like(pts)
                noise[1:-1] = np.random.normal(scale=jitter_scale, size=(segments-1, 3))
                path = pts + noise

                # теперь пробегаем каждый сегмент и делаем sphere-tracing по сегменту
                accumulated_length = 0.0
                hit_point = None
                hit_mat = None

                for seg_i in range(segments):
                    a = path[seg_i]
                    b = path[seg_i+1]
                    seg_vec = b - a
                    seg_len = np.linalg.norm(seg_vec)
                    if seg_len <= 1e-8:
                        continue
                    seg_dir = seg_vec / seg_len

                    # sample along segment using sphere-tracing like loop
                    t_seg = 0.0
                    steps = 0
                    while t_seg < seg_len and steps < max_steps_per_segment:
                        p = a + seg_dir * t_seg
                        d, _ = scene_sdf(p.reshape(1,3))
                        dval = float(d[0])
                        if dval < hit_eps:
                            # hit
                            hit_point = p
                            hit_mat = 1
                            break
                        # advance by max(dval * 0.9, hit_eps*0.5) but no more than remaining length
                        advance = max(dval * 0.9, hit_eps * 0.5)
                        if advance < 1e-5:
                            advance = hit_eps * 0.5
                        t_seg += min(advance, seg_len - t_seg)
                        steps += 1

                    accumulated_length += min(t_seg, seg_len)

                    if hit_point is not None:
                        hits += 1
                        break

                if hit_point is not None:
                    L = accumulated_length
                    amp = (1.0 / (1.0 + 0.2 * L)) * np.exp(1j * k * L)
                    pixel_amp += amp
                else:
                    # no hit: background contribution (optional phase from going to t_end)
                    L = t_end
                    amp = 0.0j  # ignore background or small contribution
                    pixel_amp += amp

            intensity = (abs(pixel_amp) ** 2) / float(spp)
            img[y, x] = intensity

            pix_idx += 1
        # печать прогресса по строкам
        done = (y+1) / height * 100
        print(f"Progress: {done:.1f}%", end='\r')

    # нормализация и гамма
    img = img / img.max()
    img = np.clip(img, 0.0, 1.0)
    img = (img ** (1.0/2.2) * 255.0).astype(np.uint8)

    im = Image.fromarray(img)
    im = im.convert('L')
    filename = 'render_path_integral_raymarcher.png'
    im.save(filename)

    print('\nDone. Saved to', filename)
    print('Elapsed', time() - start, 's')

    return filename

# ----------------------------- Запуск -----------------------------

if __name__ == '__main__':
    # Параметры по умолчанию — уменьшены для ускорения.
    # Увеличьте spp и segments для более реалистичной картины, но будет медленнее.
    render(width=220, height=140, fov=40.0, spp=60, segments=8, wavelength=0.4,
           jitter_scale=0.35, max_steps_per_segment=45, hit_eps=0.02)
