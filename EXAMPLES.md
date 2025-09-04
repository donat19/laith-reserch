# Примеры использования Path Integral Engine

## 🚀 Быстрый старт

### Базовое использование
```python
from path_integral_engine import PathIntegralEngine, PathIntegralConfig, QualityPreset

# Создаем конфигурацию для быстрого просмотра
config = PathIntegralConfig.from_preset(QualityPreset.PREVIEW)
engine = PathIntegralEngine(config)

# Простая сцена - сфера
def sphere_sdf(points):
    return torch.norm(points, dim=-1) - 1.0

# Рендерим
image, info = engine.render(
    sdf_function=sphere_sdf,
    camera_pos=[0, 0, -3],
    look_at=[0, 0, 0]
)

print(f"Рендеринг завершен: {info['fps']:.1f} FPS, {info['render_time']:.2f}с")

# Сохраняем результат
engine.save_image(image, "sphere_render.png")
```

### Интерактивное приложение
```python
import pygame
from path_integral_engine import PathIntegralEngine, PathIntegralConfig

def create_interactive_app():
    # Инициализация Pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Path Integral Raymarcher")
    
    # Настройка движка
    config = PathIntegralConfig()
    config.width = width // 2  # Уменьшаем для производительности
    config.height = height // 2
    config.adaptive_quality = True
    config.target_fps = 15.0
    
    engine = PathIntegralEngine(config)
    
    # Параметры камеры
    camera_pos = [0, 0, -5]
    camera_angles = [0, 0]  # yaw, pitch
    
    # Сцена
    def mandelbulb_sdf(points):
        """SDF для фрактала Mandelbulb"""
        z = points.clone()
        dr = 1.0
        r = torch.norm(z, dim=-1, keepdim=True)
        
        for i in range(8):  # Итерации фрактала
            # Переход в сферические координаты
            theta = torch.atan2(torch.norm(z[..., :2], dim=-1, keepdim=True), z[..., 2:3])
            phi = torch.atan2(z[..., 1:2], z[..., 0:1])
            
            # Степень 8 для Mandelbulb
            power = 8
            r_pow = torch.pow(r, power)
            theta_pow = theta * power
            phi_pow = phi * power
            
            # Обратно в декартовы координаты
            z = r_pow * torch.stack([
                torch.sin(theta_pow) * torch.cos(phi_pow),
                torch.sin(theta_pow) * torch.sin(phi_pow),
                torch.cos(theta_pow)
            ], dim=-1).squeeze()
            
            z += points  # Добавляем исходную точку
            
            # Производная для distance estimation
            dr = torch.pow(r, power - 1) * power * dr + 1.0
            r = torch.norm(z, dim=-1, keepdim=True)
            
            # Условие выхода
            if torch.all(r > 2.0):
                break
                
        return 0.5 * torch.log(r) * r / dr
    
    clock = pygame.time.Clock()
    running = True
    
    print("Управление:")
    print("  Мышь - поворот камеры")
    print("  Q/E - качество")
    print("  C - цветной/ч-б режим")
    print("  R - адаптивное качество")
    print("  SPACE - сброс камеры")
    print("  ESC - выход")
    
    while running:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_q:
                    engine.decrease_quality()
                elif event.key == pygame.K_e:
                    engine.increase_quality()
                elif event.key == pygame.K_c:
                    engine.toggle_rendering_mode()
                elif event.key == pygame.K_r:
                    engine.toggle_adaptive_quality()
                elif event.key == pygame.K_SPACE:
                    camera_pos = [0, 0, -5]
                    camera_angles = [0, 0]
        
        # Управление мышью
        if pygame.mouse.get_pressed()[0]:
            mouse_rel = pygame.mouse.get_rel()
            camera_angles[0] += mouse_rel[0] * 0.01  # yaw
            camera_angles[1] += mouse_rel[1] * 0.01  # pitch
            camera_angles[1] = max(-1.5, min(1.5, camera_angles[1]))  # Ограничиваем pitch
        
        # Вычисляем позицию камеры
        distance = 5.0
        camera_pos = [
            distance * math.sin(camera_angles[0]) * math.cos(camera_angles[1]),
            distance * math.sin(camera_angles[1]),
            distance * math.cos(camera_angles[0]) * math.cos(camera_angles[1])
        ]
        
        # Рендерим
        image, info = engine.render(
            sdf_function=mandelbulb_sdf,
            camera_pos=camera_pos,
            look_at=[0, 0, 0]
        )
        
        # Преобразуем в Pygame Surface
        if len(image.shape) == 3:  # RGB
            image_rgb = (image * 255).astype(np.uint8)
        else:  # Monochrome
            image_rgb = np.stack([image] * 3, axis=2)
            image_rgb = (image_rgb * 255).astype(np.uint8)
        
        # Масштабируем до размера экрана
        image_surface = pygame.surfarray.make_surface(image_rgb.swapaxes(0, 1))
        image_surface = pygame.transform.scale(image_surface, (width, height))
        
        # Отображаем
        screen.blit(image_surface, (0, 0))
        
        # Информация на экране
        font = pygame.font.Font(None, 36)
        info_text = f"FPS: {info['fps']:.1f} | SPP: {info['spp']} | {info['mode']}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
        clock.tick(30)  # Ограничиваем до 30 FPS дисплея
    
    pygame.quit()

if __name__ == "__main__":
    create_interactive_app()
```

## 🎨 Примеры сцен

### 1. Простые геометрические фигуры

```python
def sphere_sdf(points):
    """Сфера радиусом 1"""
    return torch.norm(points, dim=-1) - 1.0

def box_sdf(points, size=[1, 1, 1]):
    """Прямоугольный параллелепипед"""
    size = torch.tensor(size, device=points.device)
    q = torch.abs(points) - size
    return torch.norm(torch.maximum(q, torch.zeros_like(q)), dim=-1) + \
           torch.minimum(torch.max(q, dim=-1)[0], torch.zeros_like(q[..., 0]))

def cylinder_sdf(points, radius=1.0, height=1.0):
    """Цилиндр"""
    d = torch.stack([
        torch.norm(points[..., :2], dim=-1) - radius,
        torch.abs(points[..., 2]) - height
    ], dim=-1)
    return torch.minimum(torch.max(d, dim=-1)[0], 0.0) + \
           torch.norm(torch.maximum(d, 0.0), dim=-1)
```

### 2. Операции с фигурами

```python
def union_sdf(sdf1_values, sdf2_values):
    """Объединение двух SDF"""
    return torch.minimum(sdf1_values, sdf2_values)

def intersection_sdf(sdf1_values, sdf2_values):
    """Пересечение двух SDF"""
    return torch.maximum(sdf1_values, sdf2_values)

def subtraction_sdf(sdf1_values, sdf2_values):
    """Вычитание sdf2 из sdf1"""
    return torch.maximum(sdf1_values, -sdf2_values)

def smooth_union_sdf(sdf1_values, sdf2_values, k=0.1):
    """Плавное объединение"""
    h = torch.clamp(0.5 + 0.5 * (sdf2_values - sdf1_values) / k, 0.0, 1.0)
    return torch.lerp(sdf2_values, sdf1_values, h) - k * h * (1.0 - h)

# Пример композитной сцены
def complex_scene_sdf(points):
    """Сложная сцена из нескольких объектов"""
    # Основная сфера
    sphere1 = sphere_sdf(points - torch.tensor([0, 0, 0]))
    
    # Сфера для вычитания
    sphere2 = sphere_sdf(points - torch.tensor([0.5, 0.5, 0.5])) * 0.6
    
    # Цилиндр
    cyl = cylinder_sdf(points - torch.tensor([0, 0, 0]), radius=0.3, height=2.0)
    
    # Комбинируем
    result = subtraction_sdf(sphere1, sphere2)  # Сфера с дыркой
    result = smooth_union_sdf(result, cyl, k=0.2)  # Добавляем цилиндр
    
    return result
```

### 3. Фракталы

```python
def julia_set_sdf(points, c=[-0.7, 0.27015]):
    """3D Julia set"""
    z = points[..., :2]  # Используем только x,y
    c_tensor = torch.tensor(c, device=points.device)
    
    for i in range(20):  # Итерации
        z_real = z[..., 0]
        z_imag = z[..., 1]
        
        # z = z² + c
        new_real = z_real * z_real - z_imag * z_imag + c_tensor[0]
        new_imag = 2 * z_real * z_imag + c_tensor[1]
        
        z = torch.stack([new_real, new_imag], dim=-1)
        
        # Проверка на расхождение
        magnitude = torch.norm(z, dim=-1)
        if torch.all(magnitude > 2.0):
            break
    
    # Расстояние от поверхности
    return torch.norm(z, dim=-1) - 1.0

def koch_snowflake_sdf(points, iterations=3):
    """Приближенная 3D версия снежинки Коха"""
    p = points[..., :2]  # Проекция на XY плоскость
    z_dist = torch.abs(points[..., 2]) - 0.1  # Толщина
    
    # Симметрия треугольника
    p = torch.abs(p)
    if p[..., 1] + 0.577 * p[..., 0] > 1.155:
        p = torch.stack([
            p[..., 0] * 0.5 - p[..., 1] * 0.866,
            p[..., 0] * 0.866 + p[..., 1] * 0.5
        ], dim=-1)
    
    # Итерации фрактала
    for i in range(iterations):
        p = torch.abs(p - torch.tensor([0.5, 0.0]))
        if p[..., 0] + 0.577 * p[..., 1] > 0.577:
            p = torch.stack([
                p[..., 0] * 0.5 - p[..., 1] * 0.866,
                p[..., 0] * 0.866 + p[..., 1] * 0.5
            ], dim=-1)
        p *= 3.0
        p -= torch.tensor([1.0, 0.0])
    
    xy_dist = torch.norm(p, dim=-1) - 0.1
    return torch.maximum(xy_dist, z_dist)
```

### 4. Мета-объекты (Metaballs)

```python
def metaball_sdf(points, centers, radii, threshold=1.0):
    """SDF для метаболов (blended spheres)"""
    total_field = torch.zeros(points.shape[:-1], device=points.device)
    
    for center, radius in zip(centers, radii):
        center_tensor = torch.tensor(center, device=points.device)
        distance = torch.norm(points - center_tensor, dim=-1)
        # Функция влияния (обратно пропорциональна квадрату расстояния)
        field = (radius * radius) / (distance * distance + 1e-6)
        total_field += field
    
    # Преобразуем поле в SDF
    return threshold - total_field

# Пример использования
def animated_metaballs_scene(time):
    """Анимированные метаболы"""
    # Центры метаболов движутся по орбитам
    centers = [
        [2 * math.cos(time), 2 * math.sin(time), 0],
        [1.5 * math.cos(time * 1.3 + 1), 1.5 * math.sin(time * 1.3 + 1), 0.5],
        [math.cos(time * 0.7 + 2), math.sin(time * 0.7 + 2), -0.5]
    ]
    radii = [1.0, 0.8, 0.6]
    
    def scene_sdf(points):
        return metaball_sdf(points, centers, radii, threshold=2.0)
    
    return scene_sdf
```

## 🎮 Продвинутые примеры

### Анимированные сцены

```python
def create_animation_sequence():
    """Создание последовательности кадров анимации"""
    engine = PathIntegralEngine(PathIntegralConfig.from_preset(QualityPreset.HIGH))
    
    frames = []
    duration = 5.0  # секунд
    fps = 24
    total_frames = int(duration * fps)
    
    for frame_num in range(total_frames):
        time = frame_num / fps
        
        # Анимированная сцена
        scene_sdf = animated_metaballs_scene(time)
        
        # Анимированная камера
        camera_angle = time * 0.5  # Поворот вокруг сцены
        camera_pos = [
            5 * math.cos(camera_angle),
            1 + math.sin(time * 2) * 0.5,  # Вертикальные колебания
            5 * math.sin(camera_angle)
        ]
        
        # Рендерим кадр
        image, info = engine.render(
            sdf_function=scene_sdf,
            camera_pos=camera_pos,
            look_at=[0, 0, 0]
        )
        
        frames.append(image)
        
        if frame_num % 10 == 0:
            print(f"Кадр {frame_num}/{total_frames} ({frame_num/total_frames*100:.1f}%)")
    
    # Сохраняем как GIF
    save_animation_gif(frames, "metaballs_animation.gif", fps=fps)
    
    return frames

def save_animation_gif(frames, filename, fps=24):
    """Сохранение кадров как GIF анимации"""
    from PIL import Image
    
    # Преобразуем кадры в PIL Images
    pil_images = []
    for frame in frames:
        if len(frame.shape) == 2:  # Grayscale
            frame_rgb = np.stack([frame] * 3, axis=2)
        else:
            frame_rgb = frame
        
        frame_uint8 = (frame_rgb * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(frame_uint8))
    
    # Сохраняем как GIF
    pil_images[0].save(
        filename,
        save_all=True,
        append_images=pil_images[1:],
        duration=int(1000/fps),  # milliseconds per frame
        loop=0
    )
    
    print(f"Анимация сохранена: {filename}")
```

### Высококачественный рендеринг

```python
def high_quality_render(scene_sdf, output_path, resolution=(1920, 1080)):
    """Высококачественный рендеринг для финального результата"""
    
    # Максимальные настройки качества
    config = PathIntegralConfig()
    config.width = resolution[0]
    config.height = resolution[1]
    config.spp = 256  # Много семплов для минимума шума
    config.segments = 8  # Длинные пути для точности
    config.adaptive_quality = False  # Фиксированное качество
    config.jitter_scale = 0.2  # Меньше шума
    config.hit_eps = 0.01  # Высокая точность
    
    engine = PathIntegralEngine(config)
    
    # Камера для красивого ракурса
    camera_pos = [3, 2, -5]
    look_at = [0, 0, 0]
    
    print("Начинаем высококачественный рендеринг...")
    print(f"Разрешение: {resolution[0]}x{resolution[1]}")
    print(f"Семплов на пиксель: {config.spp}")
    
    start_time = time.time()
    
    # Спектральный рендеринг для максимального качества
    engine.switch_rendering_mode(RenderingMode.RGB_SPECTRAL)
    
    image, info = engine.render(
        sdf_function=scene_sdf,
        camera_pos=camera_pos,
        look_at=look_at
    )
    
    end_time = time.time()
    render_time = end_time - start_time
    
    print(f"Рендеринг завершен за {render_time:.1f} секунд")
    print(f"Средний FPS: {info['fps']:.2f}")
    
    # Постобработка для улучшения качества
    image = post_process_image(image)
    
    # Сохраняем в высоком качестве
    engine.save_image(image, output_path, quality=95)
    
    return image

def post_process_image(image):
    """Постобработка изображения"""
    # Тональная коррекция
    gamma = 2.2
    image = np.power(image, 1.0 / gamma)
    
    # Контраст и яркость
    contrast = 1.1
    brightness = 0.05
    image = image * contrast + brightness
    
    # Ограничиваем значения
    image = np.clip(image, 0.0, 1.0)
    
    return image
```

### Batch рендеринг

```python
def batch_render_scenes(scene_configs, output_dir="renders"):
    """Пакетный рендеринг множественных сцен"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Оптимальные настройки для пакетной обработки
    config = PathIntegralConfig.from_preset(QualityPreset.HIGH)
    engine = PathIntegralEngine(config)
    
    for i, scene_config in enumerate(scene_configs):
        print(f"Рендеринг сцены {i+1}/{len(scene_configs)}: {scene_config['name']}")
        
        image, info = engine.render(
            sdf_function=scene_config['sdf'],
            camera_pos=scene_config['camera_pos'],
            look_at=scene_config['look_at']
        )
        
        # Сохраняем с именем сцены
        output_path = os.path.join(output_dir, f"{scene_config['name']}.png")
        engine.save_image(image, output_path)
        
        print(f"  Сохранено: {output_path} ({info['fps']:.1f} FPS)")

# Пример использования
if __name__ == "__main__":
    # Конфигурации для пакетного рендеринга
    scenes = [
        {
            'name': 'sphere',
            'sdf': sphere_sdf,
            'camera_pos': [0, 0, -3],
            'look_at': [0, 0, 0]
        },
        {
            'name': 'complex_scene',
            'sdf': complex_scene_sdf,
            'camera_pos': [2, 1, -4],
            'look_at': [0, 0, 0]
        },
        {
            'name': 'mandelbulb',
            'sdf': mandelbulb_sdf,
            'camera_pos': [0, 0, -3],
            'look_at': [0, 0, 0]
        }
    ]
    
    batch_render_scenes(scenes)
```

## 🧪 Экспериментальные возможности

### Кастомные материалы

```python
class CustomMaterial:
    """Базовый класс для кастомных материалов"""
    
    def __init__(self, color=[1, 1, 1], roughness=0.5, metallic=0.0):
        self.color = color
        self.roughness = roughness
        self.metallic = metallic
    
    def apply_material_properties(self, hit_points, normals):
        """Применение свойств материала к точкам попадания"""
        # Базовая реализация - возвращает цвет материала
        return torch.tensor(self.color, device=hit_points.device)

def material_aware_sdf(points, materials_map):
    """SDF с информацией о материалах"""
    # Пример: разные материалы для разных объектов
    sphere_sdf_val = sphere_sdf(points)
    box_sdf_val = box_sdf(points - torch.tensor([2, 0, 0]))
    
    # Выбираем ближайший объект
    sphere_closer = sphere_sdf_val < box_sdf_val
    
    distance = torch.where(sphere_closer, sphere_sdf_val, box_sdf_val)
    material_id = torch.where(sphere_closer, 
                             torch.zeros_like(distance), 
                             torch.ones_like(distance))
    
    return distance, material_id
```

### Экспериментальные эффекты

```python
def chromatic_aberration_render(engine, scene_sdf, camera_pos, look_at):
    """Рендеринг с хроматической аберрацией"""
    
    # Рендерим каждый канал с небольшим смещением
    offsets = {
        'red': [0.01, 0, 0],
        'green': [0, 0, 0],
        'blue': [-0.01, 0, 0]
    }
    
    channels = {}
    
    for color, offset in offsets.items():
        offset_camera = [camera_pos[i] + offset[i] for i in range(3)]
        
        # Рендерим монохромно для каждого канала
        engine.switch_rendering_mode(RenderingMode.MONOCHROME)
        image, _ = engine.render(scene_sdf, offset_camera, look_at)
        
        channels[color] = image
    
    # Комбинируем каналы
    rgb_image = np.stack([
        channels['red'],
        channels['green'], 
        channels['blue']
    ], axis=2)
    
    return rgb_image

def motion_blur_render(engine, scene_sdf, camera_positions, look_at):
    """Рендеринг с motion blur"""
    
    accumulated_image = None
    
    for camera_pos in camera_positions:
        image, _ = engine.render(scene_sdf, camera_pos, look_at)
        
        if accumulated_image is None:
            accumulated_image = image.copy()
        else:
            accumulated_image += image
    
    # Усредняем
    accumulated_image /= len(camera_positions)
    
    return accumulated_image
```

Эти примеры демонстрируют полный спектр возможностей Path Integral Engine - от простых сцен до сложных анимаций и экспериментальных эффектов!
