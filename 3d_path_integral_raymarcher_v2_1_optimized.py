"""
3D Path-Integral Raymarcher v2.1 - Spectral RGB Real-time (Mac M1/M2)
----------------------------------------------------------------------
Высокооптимизированная версия для Apple Silicon с спектральным рендерингом:
- Спектральный рендеринг RGB с реалистичными длинами волн (R:650nm, G:532nm, B:450nm)
- Переключение между цветным и черно-белым режимами
- Дисперсионные эффекты для разных длин волн
- Улучшенное использование MPS (Metal Performance Shaders)
- Адаптивное качество в зависимости от FPS
- Оптимизированные вычисления для реального времени
- Интерактивное управление камерой

Требования: torch, numpy, pygame, opencv-python
"""

import math
import time
import numpy as np
import torch
import pygame
import cv2
import sys

# Настройка устройства
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class OptimizedGPURaymarcher:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        self.device = device
        
        # Адаптивные параметры качества
        self.target_fps = 15.0  # Целевой FPS
        self.spp = 16  # Начальное количество семплов
        self.min_spp = 4
        self.max_spp = 32
        
        # Параметры рендеринга
        self.fov = 45.0
        self.segments = 4  # Уменьшено для производительности
        self.jitter_scale = 0.3
        self.max_steps = 20  # Уменьшено для скорости
        self.hit_eps = 0.05
        
        # Спектральные параметры (длины волн в микрометрах)
        self.wavelengths_rgb = {
            'red': 0.650,    # Красный
            'green': 0.532,  # Зеленый
            'blue': 0.450    # Синий
        }
        
        # Режим рендеринга
        self.color_mode = True  # True - цветной, False - черно-белый
        self.monochrome_wavelength = 0.55  # Для ч/б режима
        
        # Камера
        self.cam_pos = torch.tensor([0.0, 1.0, -3.0], device=device, dtype=torch.float32)
        self.cam_target = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        self.cam_angle_x = 0.1
        self.cam_angle_y = 0.0
        
        # Pre-computed константы
        self._setup_constants()
        
        # Буферы для переиспользования
        self._init_buffers()
        
    def toggle_color_mode(self):
        """Переключение между цветным и ч/б режимами"""
        self.color_mode = not self.color_mode
        mode_name = "RGB цветной" if self.color_mode else "Черно-белый"
        print(f"🎨 Режим переключен: {mode_name}")
        
    def _setup_constants(self):
        """Предварительные вычисления констант"""
        aspect = self.width / self.height
        screen_h = 2.0 * math.tan(math.radians(self.fov) / 2.0)
        screen_w = screen_h * aspect
        
        # Координаты пикселей (постоянные)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float32),
            torch.arange(self.width, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        self.pixel_u = (x_coords + 0.5) / self.width - 0.5
        self.pixel_v = (y_coords + 0.5) / self.height - 0.5
        self.screen_u = self.pixel_u * screen_w
        self.screen_v = self.pixel_v * screen_h
        
        # Волновые числа для каждого канала
        self.k_red = 2.0 * math.pi / self.wavelengths_rgb['red']
        self.k_green = 2.0 * math.pi / self.wavelengths_rgb['green']
        self.k_blue = 2.0 * math.pi / self.wavelengths_rgb['blue']
        self.k_mono = 2.0 * math.pi / self.monochrome_wavelength
        
    def _init_buffers(self):
        """Инициализация переиспользуемых буферов"""
        self.ray_dirs = torch.zeros(self.height, self.width, 3, device=self.device)
        if self.color_mode:
            # RGB буферы для цветного режима
            self.pixel_intensities_rgb = torch.zeros(self.height, self.width, 3, device=self.device)
        else:
            # Один канал для ч/б режима
            self.pixel_intensities = torch.zeros(self.height, self.width, device=self.device)
        
    def normalize_gpu(self, v):
        """Быстрая GPU нормализация"""
        return torch.nn.functional.normalize(v, dim=-1)
        
    def update_camera(self, angle_x, angle_y):
        """Обновление камеры с орбитальным движением"""
        self.cam_angle_x = max(-math.pi/3, min(math.pi/3, angle_x))
        self.cam_angle_y = angle_y
        
        # Орбитальная камера
        radius = 4.0
        self.cam_pos = torch.tensor([
            radius * math.sin(self.cam_angle_y) * math.cos(self.cam_angle_x),
            radius * math.sin(self.cam_angle_x) + 1.0,
            radius * math.cos(self.cam_angle_y) * math.cos(self.cam_angle_x)
        ], device=self.device)
        
        # Направление камеры
        cam_dir = self.normalize_gpu(self.cam_target - self.cam_pos)
        
        # Базисные векторы
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        cam_right = self.normalize_gpu(torch.linalg.cross(cam_dir, world_up))
        cam_up = torch.linalg.cross(cam_right, cam_dir)
        
        # Обновляем направления лучей для всех пикселей
        pixel_centers = (
            self.cam_pos + cam_dir +
            cam_right * self.screen_u.unsqueeze(-1) +
            cam_up * self.screen_v.unsqueeze(-1)
        )
        
        self.ray_dirs = self.normalize_gpu(pixel_centers - self.cam_pos)
        
    def sdf_scene_optimized(self, p):
        """Оптимизированная SDF сцены"""
        # Сфера 1 (главная)
        d1 = torch.norm(p - torch.tensor([0.0, 0.0, 0.0], device=self.device), dim=-1) - 1.0
        
        # Сфера 2
        d2 = torch.norm(p - torch.tensor([2.0, 0.5, 1.0], device=self.device), dim=-1) - 0.7
        
        # Сфера 3 
        d3 = torch.norm(p - torch.tensor([-1.5, -0.3, -0.5], device=self.device), dim=-1) - 0.8
        
        # Плоскость (пол)
        d4 = p[..., 1] + 1.5
        
        # Комбинируем
        return torch.minimum(torch.minimum(torch.minimum(d1, d2), d3), d4)
        
    def adaptive_quality_control(self, current_fps):
        """Адаптивное управление качеством"""
        if current_fps < self.target_fps * 0.8:
            # Снижаем качество
            self.spp = max(self.min_spp, self.spp - 2)
        elif current_fps > self.target_fps * 1.2:
            # Повышаем качество
            self.spp = min(self.max_spp, self.spp + 1)
            
    def render_frame_optimized(self):
        """Оптимизированный рендеринг кадра с спектральным анализом"""
        start_time = time.time()
        
        if self.color_mode:
            # Цветной режим - рендерим RGB каналы отдельно
            return self._render_color_frame()
        else:
            # Черно-белый режим
            return self._render_monochrome_frame()
    
    def _render_color_frame(self):
        """Рендеринг цветного кадра"""
        start_time = time.time()
        
        # Обнуляем RGB буферы
        self.pixel_intensities_rgb = torch.zeros(self.height, self.width, 3, device=self.device)
        
        # Рендерим каждый цветовой канал отдельно
        channels = ['red', 'green', 'blue']
        k_values = [self.k_red, self.k_green, self.k_blue]
        
        for channel_idx, (channel, k_val) in enumerate(zip(channels, k_values)):
            channel_intensity = self._render_channel(k_val)
            self.pixel_intensities_rgb[:, :, channel_idx] = channel_intensity
        
        # Нормализация и гамма коррекция
        self.pixel_intensities_rgb = torch.clamp(self.pixel_intensities_rgb, 0.0, 1.0)
        gamma_corrected = torch.pow(self.pixel_intensities_rgb, 1.0/2.2)
        img_array = (gamma_corrected * 255.0).byte().cpu().numpy()
        
        render_time = time.time() - start_time
        fps = 1.0 / render_time if render_time > 0 else 0
        return img_array, fps
    
    def _render_monochrome_frame(self):
        """Рендеринг черно-белого кадра"""
        start_time = time.time()
        
        # Обнуляем буфер интенсивности
        self.pixel_intensities = torch.zeros(self.height, self.width, device=self.device)
        
        # Рендерим один канал
        self.pixel_intensities = self._render_channel(self.k_mono)
        
        # Нормализация и гамма коррекция
        self.pixel_intensities = torch.clamp(self.pixel_intensities, 0.0, 1.0)
        gamma_corrected = torch.pow(self.pixel_intensities, 1.0/2.2)
        img_array = (gamma_corrected * 255.0).byte().cpu().numpy()
        
        render_time = time.time() - start_time
        fps = 1.0 / render_time if render_time > 0 else 0
        return img_array, fps
    
    def _render_channel(self, k_value):
        """Рендеринг одного спектрального канала"""
        channel_intensity = torch.zeros(self.height, self.width, device=self.device)
        
        # Batch размер для балансировки памяти и производительности
        batch_size = min(8, self.spp)
        num_batches = (self.spp + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, self.spp)
            current_batch_size = batch_end - batch_start
            
            # Генерируем случайные смещения для path integral
            path_offsets = torch.normal(
                0, self.jitter_scale,
                size=(current_batch_size, self.height, self.width, self.segments, 3),
                device=self.device
            )
            
            # Вычисляем пути
            t_vals = torch.linspace(0.5, 8.0, self.segments, device=self.device)
            
            # Базовые позиции вдоль лучей
            base_positions = (
                self.cam_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0) +
                self.ray_dirs.unsqueeze(0).unsqueeze(3) * 
                t_vals.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            )
            
            # Добавляем смещения
            path_positions = base_positions + path_offsets
            
            # Упрощенная трассировка для данного канала
            batch_intensities = self._trace_simplified_channel(path_positions, current_batch_size, k_value)
            channel_intensity += batch_intensities.sum(dim=0)
            
        # Нормализация по количеству семплов
        channel_intensity /= self.spp
        return channel_intensity
    def _trace_simplified_channel(self, path_positions, batch_size, k_value):
        """Упрощенная трассировка для отдельного спектрального канала"""
        batch_intensities = torch.zeros(batch_size, self.height, self.width, device=self.device)
        
        for i in range(batch_size):
            positions = path_positions[i]  # [height, width, segments, 3]
            
            # Простая проверка пересечений
            min_distances = torch.full((self.height, self.width), float('inf'), device=self.device)
            optical_lengths = torch.zeros(self.height, self.width, device=self.device)
            
            for seg_idx in range(self.segments):
                pos = positions[:, :, seg_idx, :]  # [height, width, 3]
                pos_flat = pos.reshape(-1, 3)
                
                # SDF проверка
                distances = self.sdf_scene_optimized(pos_flat).reshape(self.height, self.width)
                min_distances = torch.minimum(min_distances, distances)
                
                # Аккумуляция оптической длины
                if seg_idx > 0:
                    prev_pos = positions[:, :, seg_idx-1, :]
                    segment_length = torch.norm(pos - prev_pos, dim=-1)
                    optical_lengths += segment_length
                    
            # Path integral амплитуда с учетом длины волны
            hit_mask = min_distances < self.hit_eps
            phases = k_value * optical_lengths
            
            # Спектрально-зависимые эффекты
            wavelength = 2.0 * math.pi / k_value
            
            # Дисперсионные эффекты (разные материалы по-разному преломляют свет)
            dispersion_factor = self._calculate_dispersion(wavelength)
            
            # Интенсивность с затуханием и спектральными эффектами
            attenuation = 1.0 / (1.0 + 0.05 * optical_lengths * dispersion_factor)
            
            # Интерференционные паттерны зависят от длины волны
            interference = 0.8 + 0.2 * torch.cos(phases)
            
            intensity = torch.where(
                hit_mask,
                attenuation * interference,
                torch.tensor(0.01, device=self.device)  # Небольшой фон
            )
            
            batch_intensities[i] = intensity
            
        return batch_intensities
    
    def _calculate_dispersion(self, wavelength):
        """Расчет дисперсионного фактора для данной длины волны"""
        # Упрощенная модель дисперсии (больше для коротких волн)
        return 1.0 + 0.3 * (0.55 / wavelength - 1.0)
        
    def _trace_simplified(self, path_positions, batch_size):
        """Упрощенная трассировка для производительности (для совместимости)"""
        return self._trace_simplified_channel(path_positions, batch_size, self.k_mono)

class InteractiveRenderer:
    def __init__(self, width=512, height=512):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("3D Path Integral Raymarcher v2.1 - Optimized Real-time")
        
        self.raymarcher = OptimizedGPURaymarcher(width, height)
        self.running = True
        self.mouse_pressed = False
        self.last_mouse_pos = (0, 0)
        
        # Статистика производительности
        self.fps_history = []
        self.max_fps_history = 30
        
        # UI
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
        
        # Инициализация камеры
        self.raymarcher.update_camera(0.1, 0.0)
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_pressed = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_pressed = False
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_pressed:
                    x, y = pygame.mouse.get_pos()
                    dx = x - self.last_mouse_pos[0]
                    dy = y - self.last_mouse_pos[1]
                    
                    # Обновляем углы камеры
                    self.raymarcher.cam_angle_y += dx * 0.01
                    self.raymarcher.cam_angle_x += dy * 0.01
                    
                    self.raymarcher.update_camera(
                        self.raymarcher.cam_angle_x, 
                        self.raymarcher.cam_angle_y
                    )
                    self.last_mouse_pos = (x, y)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_q:
                    # Увеличить качество
                    old_spp = self.raymarcher.spp
                    self.raymarcher.spp = min(64, self.raymarcher.spp + 4)
                    print(f"⬆️  Качество увеличено: {old_spp} → {self.raymarcher.spp} SPP")
                elif event.key == pygame.K_e:
                    # Уменьшить качество
                    old_spp = self.raymarcher.spp
                    self.raymarcher.spp = max(4, self.raymarcher.spp - 4)
                    print(f"⬇️  Качество уменьшено: {old_spp} → {self.raymarcher.spp} SPP")
                elif event.key == pygame.K_r:
                    # Переключить адаптивное качество
                    if hasattr(self, 'adaptive_quality'):
                        self.adaptive_quality = not self.adaptive_quality
                    else:
                        self.adaptive_quality = True
                    status = "ВКЛЮЧЕНО" if self.adaptive_quality else "ВЫКЛЮЧЕНО"
                    print(f"🔄 Адаптивное качество: {status}")
                elif event.key == pygame.K_c:
                    # Переключить цветной/ч-б режим
                    self.raymarcher.toggle_color_mode()
                    # Переинициализируем буферы для нового режима
                    self.raymarcher._init_buffers()
                elif event.key == pygame.K_SPACE:
                    # Сброс камеры
                    self.raymarcher.update_camera(0.1, 0.0)
                    print("📹 Камера сброшена к начальной позиции")
                        
    def update_fps_stats(self, fps):
        """Обновление статистики FPS"""
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_fps_history:
            self.fps_history.pop(0)
            
    def draw_ui(self, fps):
        """Отрисовка UI"""
        # Полупрозрачный фон для UI
        ui_surface = pygame.Surface((280, 140))
        ui_surface.set_alpha(180)
        ui_surface.fill((0, 0, 0))
        self.screen.blit(ui_surface, (10, 10))
        
        # Основная информация
        fps_text = self.font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
        spp_text = self.font.render(f"SPP: {self.raymarcher.spp}", True, (255, 255, 255))
        device_text = self.font.render(f"Device: {device.type.upper()}", True, (255, 255, 255))
        
        # Цветовой режим
        mode_color = (100, 255, 100) if self.raymarcher.color_mode else (255, 255, 100)
        mode_text = self.font.render(f"Mode: {'RGB Color' if self.raymarcher.color_mode else 'Monochrome'}", True, mode_color)
        
        self.screen.blit(fps_text, (15, 15))
        self.screen.blit(spp_text, (15, 35))
        self.screen.blit(device_text, (15, 55))
        self.screen.blit(mode_text, (15, 75))
        
        # Средний FPS
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            avg_text = self.small_font.render(f"Avg: {avg_fps:.1f}", True, (200, 200, 200))
            self.screen.blit(avg_text, (15, 95))
            
        # Статус адаптивного качества
        adaptive_status = getattr(self, 'adaptive_quality', True)
        color = (0, 255, 0) if adaptive_status else (255, 100, 100)
        adaptive_text = self.small_font.render(f"Adaptive: {'ON' if adaptive_status else 'OFF'}", True, color)
        self.screen.blit(adaptive_text, (15, 115))
            
    def run(self):
        clock = pygame.time.Clock()
        
        # Выводим управление в терминал при запуске
        print("🎮 3D Path Integral Raymarcher v2.1 - Spectral RGB")
        print("=" * 50)
        print("📱 УПРАВЛЕНИЕ:")
        print("  Mouse         - Вращение камеры")
        print("  Q / E         - Увеличить/Уменьшить качество") 
        print("  R             - Переключить адаптивное качество")
        print("  C             - Переключить цветной/ч-б режим")
        print("  Space         - Сброс камеры")
        print("  ESC           - Выход")
        print("=" * 50)
        print(f"🔧 Устройство: {device.type.upper()}")
        print(f"🌈 Режим: {'RGB' if self.raymarcher.color_mode else 'Монохромный'}")
        print("🚀 Запуск рендеринга...")
        print()
        
        self.adaptive_quality = True
        
        while self.running:
            self.handle_events()
            
            # Рендеринг
            render_start = time.time()
            img, _ = self.raymarcher.render_frame_optimized()
            render_time = time.time() - render_start
            fps = 1.0 / render_time if render_time > 0 else 0
            
            self.update_fps_stats(fps)
            
            # Адаптивное управление качеством
            if self.adaptive_quality:
                self.raymarcher.adaptive_quality_control(fps)
            
            # Конвертация в RGB для pygame
            if self.raymarcher.color_mode:
                # Уже RGB формат
                img_rgb = img
            else:
                # Конвертируем ч/б в RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
            img_surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
            
            # Отрисовка
            self.screen.fill((0, 0, 0))
            self.screen.blit(img_surface, (0, 0))
            self.draw_ui(fps)
            
            pygame.display.flip()
            clock.tick(60)  # Ограничиваем до 60 FPS для UI
            
        pygame.quit()
        print("👋 Рендерер остановлен")

def main():
    try:
        # Выбор разрешения в зависимости от устройства
        if device.type == "mps":
            width, height = 512, 512  # Полное разрешение для GPU
        else:
            width, height = 256, 256  # Пониженное разрешение для CPU
            
        renderer = InteractiveRenderer(width, height)
        renderer.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
