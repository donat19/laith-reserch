"""
3D Path-Integral Raymarcher v2.0 - Real-time GPU (Mac M1/M2 optimized)
-----------------------------------------------------------------------
Версия 2.0 с оптимизацией для Apple Silicon (M1/M2) GPU:
- Использует Metal Performance Shaders (MPS) для GPU ускорения
- Векторизованные вычисления с PyTorch + MPS backend
- Оптимизированная архитектура для real-time рендеринга
- Интерактивное окно с возможностью изменения параметров
- Adaptive sampling для повышения производительности

Требования: Python 3.8+, torch (с MPS), numpy, pygame, opencv-python
Установка: pip install torch torchvision numpy pygame opencv-python

Запуск: python 3d_path_integral_raymarcher_v2_gpu.py
"""

import math
import time
import numpy as np
import torch
import pygame
import cv2
from threading import Thread, Lock
import sys

# Проверяем доступность MPS (Metal Performance Shaders)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("⚠️  MPS not available, falling back to CPU")

class GPURaymarcher:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        self.device = device
        
        # Параметры рендеринга
        self.fov = 45.0
        self.spp = 32  # samples per pixel (уменьшено для real-time)
        self.segments = 6
        self.wavelength = 0.5
        self.jitter_scale = 0.4
        self.max_steps = 30
        self.hit_eps = 0.02
        
        # Камера
        self.cam_pos = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        self.cam_dir = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        self.cam_angle_x = 0.0
        self.cam_angle_y = 0.0
        
        # Precompute screen coordinates
        self._setup_screen_coords()
        
        # Lock for thread safety
        self.render_lock = Lock()
        
    def _setup_screen_coords(self):
        """Предварительные вычисления координат экрана для GPU"""
        aspect = self.width / self.height
        screen_h = 2.0 * math.tan(math.radians(self.fov) / 2.0)
        screen_w = screen_h * aspect
        
        # Создаем сетку координат пикселей
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float32),
            torch.arange(self.width, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Нормализованные координаты [-0.5, 0.5]
        u = (x_coords + 0.5) / self.width - 0.5
        v = (y_coords + 0.5) / self.height - 0.5
        
        self.screen_u = u * screen_w
        self.screen_v = v * screen_h
        
    def normalize_gpu(self, v):
        """GPU версия нормализации вектора"""
        norm = torch.norm(v, dim=-1, keepdim=True)
        return torch.where(norm > 1e-8, v / norm, v)
        
    def orthonormal_basis_gpu(self, forward):
        """GPU версия создания ортонормального базиса"""
        f = self.normalize_gpu(forward)
        
        # Выбираем вектор up
        up = torch.where(torch.abs(f[..., 0:1]) < 0.9,
                        torch.tensor([1.0, 0.0, 0.0], device=self.device),
                        torch.tensor([0.0, 1.0, 0.0], device=self.device))
        
        right = self.normalize_gpu(torch.cross(up, f))
        up2 = torch.cross(f, right)
        return right, up2
        
    def sdf_sphere_gpu(self, p, center, radius):
        """GPU версия SDF сферы"""
        return torch.norm(p - center, dim=-1) - radius
        
    def sdf_plane_gpu(self, p, normal, d):
        """GPU версия SDF плоскости"""
        return torch.sum(p * normal, dim=-1) + d
        
    def scene_sdf_gpu(self, p):
        """GPU версия SDF сцены с батчированными вычислениями"""
        # Сфера 1
        d1 = self.sdf_sphere_gpu(p, torch.tensor([0.0, 0.5, 4.0], device=self.device), 0.8)
        # Сфера 2  
        d2 = self.sdf_sphere_gpu(p, torch.tensor([1.5, -0.3, 5.5], device=self.device), 0.7)
        # Сфера 3 (добавляем для интересности)
        d3 = self.sdf_sphere_gpu(p, torch.tensor([-1.2, 0.8, 6.0], device=self.device), 0.5)
        # Плоскость (пол)
        d4 = self.sdf_plane_gpu(p, torch.tensor([0.0, 1.0, 0.0], device=self.device), 0.8)
        
        # Комбинируем все объекты
        d = torch.minimum(torch.minimum(torch.minimum(d1, d2), d3), d4)
        return d
        
    def update_camera(self, angle_x, angle_y):
        """Обновление позиции камеры"""
        self.cam_angle_x = angle_x
        self.cam_angle_y = angle_y
        
        # Вращение камеры
        cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)
        cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
        
        self.cam_dir = torch.tensor([
            sin_y * cos_x,
            sin_x,
            cos_y * cos_x
        ], device=self.device, dtype=torch.float32)
        
        # Позиция камеры (орбитальная)
        radius = 3.0
        self.cam_pos = torch.tensor([
            -radius * sin_y * cos_x,
            -radius * sin_x,
            -radius * cos_y * cos_x
        ], device=self.device, dtype=torch.float32)
        
    def render_frame_gpu(self):
        """Основная функция рендеринга на GPU"""
        with self.render_lock:
            start_time = time.time()
            
            # Получаем базис камеры
            right, up = self.orthonormal_basis_gpu(self.cam_dir)
            
            # Вычисляем направления лучей для всех пикселей
            pixel_centers = (
                self.cam_pos.unsqueeze(0).unsqueeze(0) + 
                self.cam_dir.unsqueeze(0).unsqueeze(0) +
                right.unsqueeze(0).unsqueeze(0) * self.screen_u.unsqueeze(-1) +
                up.unsqueeze(0).unsqueeze(0) * self.screen_v.unsqueeze(-1)
            )
            
            primary_dirs = self.normalize_gpu(pixel_centers - self.cam_pos.unsqueeze(0).unsqueeze(0))
            
            # Волновое число
            k = 2.0 * math.pi / self.wavelength
            
            # Инициализация амплитуд пикселей
            pixel_amps = torch.zeros(self.height, self.width, dtype=torch.complex64, device=self.device)
            
            # Векторизованный рендеринг с батчами семплов
            batch_size = 8  # Обрабатываем по 8 семплов за раз для экономии памяти
            for batch_start in range(0, self.spp, batch_size):
                batch_end = min(batch_start + batch_size, self.spp)
                current_batch_size = batch_end - batch_start
                
                # Генерируем случайные пути для батча
                t_end = 20.0
                t_vals = torch.linspace(0.0, t_end, self.segments + 1, device=self.device)
                
                # Базовые точки путей [batch, height, width, segments+1, 3]
                batch_dirs = primary_dirs.unsqueeze(0).expand(current_batch_size, -1, -1, -1)
                batch_origins = self.cam_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
                    current_batch_size, self.height, self.width, 1, -1)
                
                pts = (batch_origins + 
                       batch_dirs.unsqueeze(3) * t_vals.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1))
                
                # Добавляем случайный шум
                noise = torch.zeros_like(pts)
                noise[:, :, :, 1:-1, :] = torch.normal(
                    0, self.jitter_scale, 
                    size=(current_batch_size, self.height, self.width, self.segments-1, 3),
                    device=self.device
                )
                path_points = pts + noise
                
                # Трассировка лучей по сегментам
                batch_amps = self.trace_paths_vectorized(path_points, k)
                pixel_amps += batch_amps.sum(dim=0)
                
            # Вычисляем интенсивность
            intensities = torch.abs(pixel_amps) ** 2 / self.spp
            
            # Нормализация и гамма-коррекция
            intensities = intensities / (intensities.max() + 1e-8)
            intensities = torch.clamp(intensities, 0.0, 1.0)
            img = (intensities ** (1.0/2.2) * 255.0).byte()
            
            render_time = time.time() - start_time
            fps = 1.0 / render_time if render_time > 0 else 0
            
            return img.cpu().numpy(), fps
            
    def trace_paths_vectorized(self, path_points, k):
        """Векторизованная трассировка путей на GPU"""
        batch_size, height, width, num_points, _ = path_points.shape
        
        # Инициализация накопленных длин и амплитуд
        accumulated_lengths = torch.zeros(batch_size, height, width, device=self.device)
        hit_flags = torch.zeros(batch_size, height, width, dtype=torch.bool, device=self.device)
        
        # Обработка каждого сегмента
        for seg_i in range(num_points - 1):
            if hit_flags.all():
                break
                
            a = path_points[:, :, :, seg_i, :]
            b = path_points[:, :, :, seg_i + 1, :]
            seg_vec = b - a
            seg_len = torch.norm(seg_vec, dim=-1)
            
            # Маска для валидных сегментов
            valid_mask = (seg_len > 1e-8) & (~hit_flags)
            
            if not valid_mask.any():
                continue
                
            seg_dir = torch.where(valid_mask.unsqueeze(-1), 
                                seg_vec / seg_len.unsqueeze(-1).clamp(min=1e-8), 
                                torch.zeros_like(seg_vec))
            
            # Упрощенная трассировка по сегменту (меньше итераций для real-time)
            t_seg = torch.zeros_like(seg_len)
            for step in range(min(self.max_steps, 15)):  # Ограничиваем для скорости
                current_pos = a + seg_dir * t_seg.unsqueeze(-1)
                
                # Reshape для SDF вычислений
                pos_flat = current_pos[valid_mask]
                if len(pos_flat) == 0:
                    break
                    
                d_flat = self.scene_sdf_gpu(pos_flat)
                
                # Проверка попаданий
                hit_mask_flat = d_flat < self.hit_eps
                if hit_mask_flat.any():
                    # Обновляем флаги попаданий
                    full_hit_mask = torch.zeros_like(valid_mask)
                    full_hit_mask[valid_mask] = hit_mask_flat
                    hit_flags = hit_flags | full_hit_mask
                
                # Продвижение по лучу
                advance = torch.clamp(d_flat * 0.8, min=self.hit_eps * 0.5)
                
                # Обновляем t_seg только для валидных позиций
                advance_full = torch.zeros_like(t_seg)
                advance_full[valid_mask] = advance
                t_seg = torch.clamp(t_seg + advance_full, max=seg_len)
                
                # Проверяем, достигли ли конца сегмента
                end_mask = t_seg >= seg_len * 0.99
                valid_mask = valid_mask & (~end_mask)
                
                if not valid_mask.any():
                    break
            
            # Накапливаем длины
            accumulated_lengths += torch.minimum(t_seg, seg_len)
        
        # Вычисляем амплитуды
        phases = k * accumulated_lengths
        attenuation = 1.0 / (1.0 + 0.1 * accumulated_lengths)
        
        # Добавляем небольшую фоновую составляющую для неhit лучей
        bg_contribution = torch.where(hit_flags, 
                                    torch.complex(attenuation * torch.cos(phases), 
                                                attenuation * torch.sin(phases)),
                                    torch.complex(torch.tensor(0.02), torch.tensor(0.0)))
        
        return bg_contribution

class RealTimeRenderer:
    def __init__(self, width=512, height=512):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("3D Path Integral Raymarcher v2.0 - GPU Real-time")
        
        self.raymarcher = GPURaymarcher(width, height)
        self.running = True
        self.mouse_pressed = False
        self.last_mouse_pos = (0, 0)
        
        # UI элементы
        self.font = pygame.font.Font(None, 24)
        self.fps_display = 0
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
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
                    self.raymarcher.cam_angle_x = max(-math.pi/2, min(math.pi/2, self.raymarcher.cam_angle_x))
                    
                    self.raymarcher.update_camera(self.raymarcher.cam_angle_x, self.raymarcher.cam_angle_y)
                    self.last_mouse_pos = (x, y)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Сброс камеры
                    self.raymarcher.update_camera(0, 0)
                elif event.key == pygame.K_1:
                    # Уменьшить качество для скорости
                    self.raymarcher.spp = max(8, self.raymarcher.spp - 4)
                elif event.key == pygame.K_2:
                    # Увеличить качество
                    self.raymarcher.spp = min(64, self.raymarcher.spp + 4)
                    
    def draw_ui(self):
        """Отрисовка UI элементов"""
        fps_text = self.font.render(f"FPS: {self.fps_display:.1f}", True, (255, 255, 255))
        spp_text = self.font.render(f"SPP: {self.raymarcher.spp}", True, (255, 255, 255))
        device_text = self.font.render(f"Device: {device.type}", True, (255, 255, 255))
        
        # Фон для текста
        pygame.draw.rect(self.screen, (0, 0, 0, 128), (5, 5, 200, 80))
        
        self.screen.blit(fps_text, (10, 10))
        self.screen.blit(spp_text, (10, 35))
        self.screen.blit(device_text, (10, 60))
        
        # Инструкции
        help_texts = [
            "Mouse: Rotate camera",
            "Space: Reset camera",
            "1/2: Quality -/+",
            "ESC: Exit"
        ]
        
        for i, text in enumerate(help_texts):
            help_surface = self.font.render(text, True, (200, 200, 200))
            self.screen.blit(help_surface, (10, self.height - 100 + i * 20))
    
    def run(self):
        clock = pygame.time.Clock()
        
        print("🎮 Real-time Raymarcher v2.0 started!")
        print("Controls:")
        print("  Mouse: Rotate camera")
        print("  Space: Reset camera") 
        print("  1/2: Decrease/Increase quality")
        print("  ESC: Exit")
        
        while self.running:
            self.handle_events()
            
            # Рендеринг кадра
            img, fps = self.raymarcher.render_frame_gpu()
            self.fps_display = fps
            
            # Конвертация в RGB для pygame
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
            
            # Отрисовка
            self.screen.blit(img_surface, (0, 0))
            self.draw_ui()
            
            pygame.display.flip()
            clock.tick(60)  # Ограничиваем до 60 FPS
        
        pygame.quit()

def main():
    print("🚀 3D Path Integral Raymarcher v2.0 - GPU Real-time")
    print(f"Device: {device}")
    
    # Проверяем, что PyTorch видит MPS
    if device.type == "mps":
        print("✅ Apple Silicon GPU acceleration enabled!")
    else:
        print("⚠️  Running on CPU - performance may be limited")
    
    try:
        renderer = RealTimeRenderer(width=512, height=512)
        renderer.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
