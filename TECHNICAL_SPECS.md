# Технические спецификации Path Integral Engine

## 📋 Обзор архитектуры

### Файловая структура проекта
```
laith-reserch/
├── 3_d_path_integral_raymarcher.py          # v1.0 - Базовая CPU версия
├── 3d_path_integral_raymarcher_v2_gpu.py    # v2.0 - Первая GPU версия
├── 3d_path_integral_raymarcher_v2_1_optimized.py  # v2.1 - Спектральная версия
├── path_integral_engine.py                  # v3.0 - Унифицированный движок
├── engine_demo.py                          # Демонстрация движка v3.0
├── README.md                               # Основная документация
├── TECHNICAL_SPECS.md                      # Этот файл
└── render_path_integral_raymarcher.png     # Результат v1.0
```

## 🔧 Технические детали реализации

### Алгоритм Path Integral Raymarching

#### 1. Генерация путей (Path Generation)
```python
def generate_path(ray_origin, ray_direction, segments, jitter_scale):
    """
    Генерация случайного пути света для Path Integral
    
    Параметры:
    - ray_origin: начальная точка (камера)
    - ray_direction: направление луча
    - segments: количество сегментов пути
    - jitter_scale: масштаб случайных отклонений
    
    Возвращает: массив точек пути
    """
    # Базовый прямолинейный путь
    t_values = linspace(0, max_distance, segments + 1)
    base_path = ray_origin + ray_direction * t_values
    
    # Случайные отклонения (квантовая неопределенность)
    noise = normal(0, jitter_scale, size=(segments-1, 3))
    path[1:-1] += noise  # Не изменяем начало и конец
    
    return path
```

#### 2. Sphere Tracing по сегментам
```python
def trace_segment(start_point, end_point, scene_sdf, max_steps, hit_epsilon):
    """
    Трассировка одного сегмента пути через SDF сцену
    
    Возвращает: (hit_found, hit_point, distance_traveled)
    """
    segment_vector = end_point - start_point
    segment_length = norm(segment_vector)
    segment_direction = segment_vector / segment_length
    
    t = 0.0
    for step in range(max_steps):
        current_point = start_point + segment_direction * t
        distance = scene_sdf(current_point)
        
        if distance < hit_epsilon:
            return True, current_point, t
            
        # Продвигаемся на безопасное расстояние
        advance = max(distance * safety_factor, min_advance)
        t += min(advance, segment_length - t)
        
        if t >= segment_length:
            break
            
    return False, None, segment_length
```

#### 3. Вычисление квантовых амплитуд
```python
def compute_amplitude(path, wavelength, hit_point=None):
    """
    Вычисление квантовой амплитуды для данного пути
    
    Основано на принципе наименьшего действия Фейнмана
    """
    # Оптическая длина пути
    optical_length = compute_path_length(path)
    
    # Волновое число
    k = 2π / wavelength
    
    # Фаза (действие в единицах ℏ)
    phase = k * optical_length
    
    # Амплитуда с затуханием
    attenuation = 1.0 / (1.0 + attenuation_factor * optical_length)
    
    if hit_point is not None:
        # Интерференционные эффекты при попадании
        amplitude = attenuation * exp(1j * phase)
    else:
        # Фоновая составляющая
        amplitude = background_factor * attenuation
        
    return amplitude
```

#### 4. Суммирование и получение интенсивности
```python
def pixel_intensity(pixel_coord, camera, scene_sdf, samples_per_pixel):
    """
    Вычисление интенсивности пикселя методом Монте-Карло
    """
    total_amplitude = 0.0 + 0.0j
    
    for sample in range(samples_per_pixel):
        # Генерируем случайный путь
        path = generate_random_path(pixel_coord, camera)
        
        # Трассируем путь через сцену
        optical_length, hit_info = trace_path(path, scene_sdf)
        
        # Вычисляем амплитуду
        amplitude = compute_amplitude(path, wavelength, hit_info)
        
        # Накапливаем
        total_amplitude += amplitude
    
    # Интенсивность = |амплитуда|²
    intensity = abs(total_amplitude / samples_per_pixel) ** 2
    
    return intensity
```

### Спектральный рендеринг (v2.1+)

#### RGB каналы с физическими длинами волн
```python
WAVELENGTHS = {
    'red':   650e-9,  # 650 нм (красный)
    'green': 532e-9,  # 532 нм (зеленый лазер)
    'blue':  450e-9   # 450 нм (синий)
}

def render_spectral_frame():
    """Рендеринг с отдельными RGB каналами"""
    rgb_buffer = zeros((height, width, 3))
    
    for i, (color, wavelength) in enumerate(WAVELENGTHS.items()):
        # Рендерим каждый канал отдельно
        channel_intensity = render_channel(wavelength)
        
        # Применяем дисперсионные эффекты
        dispersion = compute_dispersion_factor(wavelength)
        channel_intensity *= dispersion
        
        rgb_buffer[:, :, i] = channel_intensity
    
    return rgb_buffer
```

#### Дисперсионные эффекты
```python
def compute_dispersion_factor(wavelength):
    """
    Модель дисперсии материала (упрощенная формула Коши)
    
    n(λ) = A + B/λ² + C/λ⁴ + ...
    где n - показатель преломления
    """
    # Референсная длина волны (желто-зеленый)
    lambda_ref = 550e-9
    
    # Коэффициент дисперсии (больше для коротких волн)
    dispersion_strength = 0.3
    
    factor = 1.0 + dispersion_strength * (lambda_ref / wavelength - 1.0)
    
    return factor
```

### GPU оптимизации (MPS/CUDA)

#### Векторизованные вычисления
```python
def vectorized_path_tracing(ray_dirs, camera_pos, spp, segments):
    """
    Векторизованная трассировка путей на GPU
    
    Обрабатывает множественные лучи и семплы одновременно
    """
    # Размерности: [batch, height, width, segments, 3]
    batch_size = min(8, spp)  # Ограничиваем для экономии памяти
    
    for batch_start in range(0, spp, batch_size):
        batch_end = min(batch_start + batch_size, spp)
        current_batch_size = batch_end - batch_start
        
        # Генерируем пути для всего батча
        path_positions = generate_batch_paths(
            ray_dirs, camera_pos, current_batch_size, segments
        )
        
        # Параллельная трассировка
        batch_intensities = trace_batch_paths(path_positions)
        
        # Накапливаем результат
        pixel_intensities += batch_intensities.sum(dim=0)
    
    return pixel_intensities / spp
```

#### Оптимизация памяти GPU
```python
class GPUMemoryManager:
    """Управление памятью GPU для больших разрешений"""
    
    def __init__(self, device, max_memory_gb=8):
        self.device = device
        self.max_memory = max_memory_gb * 1024**3  # В байтах
        
    def estimate_memory_usage(self, width, height, spp, segments):
        """Оценка использования памяти"""
        # Размер одного float32 тензора
        bytes_per_element = 4
        
        # Основные буферы
        ray_buffer = width * height * 3 * bytes_per_element
        path_buffer = spp * width * height * segments * 3 * bytes_per_element
        intensity_buffer = width * height * bytes_per_element
        
        total_memory = ray_buffer + path_buffer + intensity_buffer
        
        return total_memory
        
    def optimize_batch_size(self, width, height, spp, segments):
        """Автоматическое определение оптимального размера батча"""
        memory_per_sample = self.estimate_memory_usage(width, height, 1, segments)
        max_samples = self.max_memory // memory_per_sample
        
        optimal_batch_size = min(max_samples, spp, 16)  # Максимум 16 для эффективности
        
        return max(1, optimal_batch_size)
```

### Адаптивное управление качеством

#### Алгоритм адаптации
```python
class AdaptiveQualityController:
    """Контроллер адаптивного качества"""
    
    def __init__(self, target_fps=15.0, min_spp=4, max_spp=64):
        self.target_fps = target_fps
        self.min_spp = min_spp
        self.max_spp = max_spp
        
        # Параметры ПИД-регулятора
        self.kp = 0.5  # Пропорциональный коэффициент
        self.ki = 0.1  # Интегральный коэффициент
        self.kd = 0.2  # Дифференциальный коэффициент
        
        self.error_history = []
        self.last_error = 0.0
        
    def update_quality(self, current_fps, current_spp):
        """Обновление качества на основе текущего FPS"""
        # Ошибка = желаемый FPS - текущий FPS
        error = self.target_fps - current_fps
        
        # ПИД-регулятор
        proportional = self.kp * error
        
        self.error_history.append(error)
        if len(self.error_history) > 10:
            self.error_history.pop(0)
        integral = self.ki * sum(self.error_history)
        
        derivative = self.kd * (error - self.last_error)
        self.last_error = error
        
        # Вычисляем изменение SPP
        pid_output = proportional + integral + derivative
        spp_delta = int(round(pid_output))
        
        # Применяем изменение с ограничениями
        new_spp = max(self.min_spp, min(self.max_spp, current_spp + spp_delta))
        
        return new_spp
```

## 📊 Профилирование производительности

### Измерение времени рендеринга
```python
class PerformanceProfiler:
    """Профайлер производительности рендеринга"""
    
    def __init__(self):
        self.timings = {}
        self.counters = {}
        
    def time_function(self, name):
        """Декоратор для измерения времени выполнения"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                if name not in self.timings:
                    self.timings[name] = []
                self.timings[name].append(end_time - start_time)
                
                return result
            return wrapper
        return decorator
        
    def get_stats(self):
        """Получение статистики производительности"""
        stats = {}
        for name, times in self.timings.items():
            stats[name] = {
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'std_time': np.std(times),
                'total_calls': len(times)
            }
        return stats
```

### Bottleneck анализ
```python
def analyze_bottlenecks(width, height, spp, segments):
    """Анализ узких мест производительности"""
    
    profiler = PerformanceProfiler()
    
    # Измеряем различные этапы
    @profiler.time_function('path_generation')
    def timed_path_generation():
        return generate_paths(width, height, spp, segments)
    
    @profiler.time_function('sdf_evaluation') 
    def timed_sdf_evaluation(paths):
        return evaluate_sdf_batch(paths)
    
    @profiler.time_function('amplitude_computation')
    def timed_amplitude_computation(paths, sdf_results):
        return compute_amplitudes_batch(paths, sdf_results)
    
    @profiler.time_function('intensity_accumulation')
    def timed_intensity_accumulation(amplitudes):
        return accumulate_intensities(amplitudes)
    
    # Выполняем полный цикл рендеринга
    paths = timed_path_generation()
    sdf_results = timed_sdf_evaluation(paths)
    amplitudes = timed_amplitude_computation(paths, sdf_results)
    intensities = timed_intensity_accumulation(amplitudes)
    
    # Анализируем результаты
    stats = profiler.get_stats()
    
    print("🔍 Анализ производительности:")
    for stage, timing in stats.items():
        percentage = (timing['avg_time'] / sum(t['avg_time'] for t in stats.values())) * 100
        print(f"  {stage}: {timing['avg_time']:.3f}с ({percentage:.1f}%)")
    
    return stats
```

## 🧪 Тестирование и валидация

### Юнит-тесты
```python
import unittest

class TestPathIntegralEngine(unittest.TestCase):
    """Тесты для движка Path Integral"""
    
    def setUp(self):
        self.config = PathIntegralConfig()
        self.config.width = 64
        self.config.height = 64
        self.config.spp = 4
        self.engine = PathIntegralEngine(self.config)
        
    def test_engine_initialization(self):
        """Тест инициализации движка"""
        self.assertIsNotNone(self.engine.backend)
        self.assertEqual(self.engine.config.width, 64)
        
    def test_simple_render(self):
        """Тест простого рендеринга"""
        def simple_sdf(points):
            # Простая сфера
            return torch.norm(points, dim=-1) - 1.0
            
        img, info = self.engine.render(
            simple_sdf, 
            [0, 0, -3], 
            [0, 0, 0]
        )
        
        self.assertEqual(img.shape, (64, 64))
        self.assertGreater(info['fps'], 0)
        
    def test_spectral_rendering(self):
        """Тест спектрального рендеринга"""
        self.engine.switch_rendering_mode(RenderingMode.RGB_SPECTRAL)
        
        def simple_sdf(points):
            return torch.norm(points, dim=-1) - 1.0
            
        img, info = self.engine.render(
            simple_sdf,
            [0, 0, -3],
            [0, 0, 0]
        )
        
        self.assertEqual(len(img.shape), 3)  # RGB image
        self.assertEqual(img.shape[2], 3)    # 3 channels
        
    def test_quality_presets(self):
        """Тест предустановок качества"""
        for preset in [QualityPreset.PREVIEW, QualityPreset.INTERACTIVE]:
            config = PathIntegralConfig.from_preset(preset)
            engine = PathIntegralEngine(config)
            
            self.assertIsNotNone(engine.backend)
            self.assertGreater(config.spp, 0)
```

### Валидация результатов
```python
def validate_physical_correctness():
    """Проверка физической корректности результатов"""
    
    # Тест 1: Энергосохранение
    def test_energy_conservation():
        """Суммарная интенсивность не должна превышать входящую энергию"""
        # Реализация теста...
        pass
    
    # Тест 2: Интерференционные паттерны
    def test_interference_patterns():
        """Проверка корректности интерференционных эффектов"""
        # Создаем сцену с двумя когерентными источниками
        # Проверяем наличие интерференционных полос
        pass
    
    # Тест 3: Дисперсионные эффекты
    def test_dispersion_effects():
        """Проверка зависимости от длины волны"""
        wavelengths = [450e-9, 550e-9, 650e-9]  # Синий, зеленый, красный
        results = []
        
        for wavelength in wavelengths:
            # Рендеринг с различными длинами волн
            # Короткие волны должны больше рассеиваться
            pass
```

## 🔧 Настройка и оптимизация

### Рекомендуемые настройки по устройствам

#### Apple Silicon (M1/M2/M3)
```python
def get_apple_silicon_config():
    config = PathIntegralConfig()
    config.backend = RenderingBackend.GPU_MPS
    config.width = 512
    config.height = 512
    config.spp = 16
    config.segments = 4
    config.target_fps = 15.0
    config.adaptive_quality = True
    return config
```

#### Intel Mac / PC (CPU)
```python
def get_cpu_config():
    config = PathIntegralConfig()
    config.backend = RenderingBackend.CPU_NUMPY
    config.width = 256
    config.height = 256
    config.spp = 8
    config.segments = 3
    config.target_fps = 5.0
    config.adaptive_quality = True
    return config
```

#### NVIDIA GPU (будущая поддержка)
```python
def get_cuda_config():
    config = PathIntegralConfig()
    config.backend = RenderingBackend.GPU_CUDA
    config.width = 1024
    config.height = 1024
    config.spp = 32
    config.segments = 6
    config.target_fps = 10.0
    config.adaptive_quality = True
    return config
```

### Параметры тонкой настройки

```python
# Физические параметры
config.jitter_scale = 0.3      # Квантовая неопределенность (0.1-0.5)
config.hit_eps = 0.05          # Точность пересечений (0.01-0.1)
config.max_steps = 20          # Максимум шагов sphere tracing (10-50)

# Длины волн (в микрометрах)
config.wavelengths = {
    'red': 0.650,      # Стандартный красный
    'green': 0.532,    # Зеленый лазер
    'blue': 0.450,     # Стандартный синий
    'monochrome': 0.55 # Желто-зеленый для ч/б
}

# Адаптивные параметры
config.target_fps = 15.0       # Целевой FPS (5-30)
config.min_spp = 4             # Минимум семплов (1-8)
config.max_spp = 64            # Максимум семплов (16-256)
```

## 📈 Планы развития

### Версия 3.1
- [ ] CUDA backend для NVIDIA GPU
- [ ] Mesh геометрия (OBJ/PLY файлы)
- [ ] PBR материалы
- [ ] Volumetric рендеринг

### Версия 3.2
- [ ] Neural denoising (AI снижение шума)
- [ ] Temporal accumulation (накопление по времени)
- [ ] Multi-GPU поддержка
- [ ] OpenGL интеграция

### Версия 4.0
- [ ] VR/AR поддержка
- [ ] Real-time global illumination
- [ ] Procedural материалы
- [ ] Cloud рендеринг
