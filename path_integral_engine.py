"""
Path Integral Rendering Engine v3.0
====================================
Унифицированный движок рендеринга на основе интегралов по путям
Объединяет все предыдущие версии в единую модульную архитектуру

Автор: AI Assistant
Дата: 4 сентября 2025
"""

import math
import time
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np


class RenderingBackend(Enum):
    """Типы backend'ов для рендеринга"""
    CPU_NUMPY = "cpu_numpy"
    GPU_MPS = "gpu_mps" 
    GPU_CUDA = "gpu_cuda"
    CPU_TORCH = "cpu_torch"


class RenderingMode(Enum):
    """Режимы рендеринга"""
    MONOCHROME = "monochrome"
    RGB_SPECTRAL = "rgb_spectral"
    RGB_STANDARD = "rgb_standard"


class QualityPreset(Enum):
    """Предустановки качества"""
    PREVIEW = "preview"      # Быстрый предпросмотр
    INTERACTIVE = "interactive"  # Интерактивный режим
    PRODUCTION = "production"    # Высокое качество
    RESEARCH = "research"        # Максимальное качество


class PathIntegralConfig:
    """Конфигурация для Path Integral рендеринга"""
    
    def __init__(self):
        # Основные параметры
        self.width: int = 512
        self.height: int = 512
        self.spp: int = 16  # Samples per pixel
        self.segments: int = 4  # Сегменты пути
        self.fov: float = 45.0
        
        # Физические параметры
        self.wavelengths = {
            'red': 0.650,    # микрометры
            'green': 0.532,
            'blue': 0.450,
            'monochrome': 0.55
        }
        
        # Параметры трассировки
        self.jitter_scale: float = 0.3
        self.hit_eps: float = 0.05
        self.max_steps: int = 20
        self.max_distance: float = 50.0
        
        # Адаптивные параметры
        self.target_fps: float = 15.0
        self.min_spp: int = 4
        self.max_spp: int = 64
        self.adaptive_quality: bool = True
        
        # Режимы
        self.rendering_mode: RenderingMode = RenderingMode.RGB_SPECTRAL
        self.backend: RenderingBackend = RenderingBackend.GPU_MPS
        
        # Система нормализации частот фотонов
        self.enable_normalization: bool = False
        self.normalization_frames: int = 100
        self.normalization_strength: float = 1.0  # Сила применения нормализации (0.0-1.0)
        
    @classmethod
    def from_preset(cls, preset: QualityPreset) -> 'PathIntegralConfig':
        """Создание конфигурации из предустановки"""
        config = cls()
        
        if preset == QualityPreset.PREVIEW:
            config.width, config.height = 256, 256
            config.spp = 4
            config.segments = 2
            config.target_fps = 30.0
            
        elif preset == QualityPreset.INTERACTIVE:
            config.width, config.height = 512, 512
            config.spp = 16
            config.segments = 4
            config.target_fps = 15.0
            
        elif preset == QualityPreset.PRODUCTION:
            config.width, config.height = 1024, 1024
            config.spp = 64
            config.segments = 8
            config.target_fps = 1.0
            config.adaptive_quality = False
            
        elif preset == QualityPreset.RESEARCH:
            config.width, config.height = 1024, 1024
            config.spp = 256
            config.segments = 16
            config.target_fps = 0.1
            config.adaptive_quality = False
            
        return config


class PhotonFrequencyMap:
    """Система нормализации частот колебаний фотонов с интерференцией"""
    
    def __init__(self, width: int, height: int, normalization_frames: int = 100, device=None):
        self.width = width
        self.height = height
        self.normalization_frames = normalization_frames
        self.device = device or torch.device('cpu')
        
        # Карты накопления данных
        self.amplitude_map = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        self.frequency_map = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        self.phase_map = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        self.frame_count = 0
        
        # Карты интерференции для RGB каналов
        self.rgb_interference_maps = {
            'red': torch.zeros((height, width), dtype=torch.complex64, device=self.device),
            'green': torch.zeros((height, width), dtype=torch.complex64, device=self.device),
            'blue': torch.zeros((height, width), dtype=torch.complex64, device=self.device)
        }
        
        # Нормализованная карта интенсивности
        self.normalized_intensity = torch.zeros((height, width, 3), dtype=torch.float32, device=self.device)
        self.is_normalized = False
        
    def accumulate_frame(self, raw_amplitudes: Dict[str, torch.Tensor], wavelengths: Dict[str, float]):
        """Накопление данных амплитуд и фаз за один кадр"""
        if self.frame_count >= self.normalization_frames:
            return
            
        # Накапливаем комплексные амплитуды для каждого канала
        for channel, amplitude in raw_amplitudes.items():
            if channel in self.rgb_interference_maps:
                wavelength = wavelengths[channel]
                
                # Вычисляем частоту: f = c / λ
                c = 3e8  # скорость света м/с
                frequency = c / (wavelength * 1e-6)  # частота в Гц
                
                # Создаем комплексную амплитуду с фазой
                phase = 2 * math.pi * frequency * self.frame_count / 60.0  # Предполагаем 60 FPS
                
                # Если amplitude уже комплексная, берем модуль для действительной части
                if torch.is_complex(amplitude):
                    amplitude_real = torch.abs(amplitude)
                else:
                    amplitude_real = amplitude
                
                # Создаем комплексные компоненты на том же устройстве
                phase_tensor = torch.tensor(phase, device=amplitude.device, dtype=torch.float32)
                real_part = amplitude_real * torch.cos(phase_tensor)
                imag_part = amplitude_real * torch.sin(phase_tensor)
                complex_amplitude = torch.complex(real_part, imag_part)
                
                # Переносим на устройство карты интерференции
                complex_amplitude = complex_amplitude.to(self.rgb_interference_maps[channel].device)
                
                # Накапливаем в карте интерференции
                self.rgb_interference_maps[channel] += complex_amplitude
                
        self.frame_count += 1
        
        # Если достигли 100 кадров, вычисляем нормализацию
        if self.frame_count == self.normalization_frames:
            self._compute_normalization()
    
    def _compute_normalization(self):
        """Вычисление нормализованной карты с учетом интерференции"""
        print(f"🔬 Вычисление карты интерференции для {self.normalization_frames} кадров...")
        
        # Для каждого канала вычисляем интерференционную картину
        for i, (channel, complex_map) in enumerate(self.rgb_interference_maps.items()):
            # Интенсивность = |амплитуда|²
            intensity = torch.abs(complex_map) ** 2
            
            # Нормализуем интенсивность
            max_intensity = torch.max(intensity)
            if max_intensity > 0:
                intensity = intensity / max_intensity
            
            self.normalized_intensity[:, :, i] = intensity
        
        # Вычисляем глобальную карту частот (средняя по всем каналам)
        total_amplitude = torch.abs(
            self.rgb_interference_maps['red'] + 
            self.rgb_interference_maps['green'] + 
            self.rgb_interference_maps['blue']
        )
        
        # Нормализуем общую амплитуду
        max_amp = torch.max(total_amplitude)
        if max_amp > 0:
            self.amplitude_map = total_amplitude / max_amp
        
        self.is_normalized = True
        print("✅ Карта интерференции готова!")
    
    def apply_normalization(self, raw_image: torch.Tensor) -> torch.Tensor:
        """Применение нормализации к изображению"""
        if not self.is_normalized:
            # Если нормализация ещё не готова, возвращаем исходное изображение
            return raw_image
        
        # Переносим normalized_intensity на устройство изображения
        normalized_intensity = self.normalized_intensity.to(raw_image.device)
        
        # Применяем интерференционную коррекцию
        if len(raw_image.shape) == 3:  # RGB изображение
            corrected_image = raw_image * normalized_intensity
        else:  # Монохромное изображение
            avg_intensity = torch.mean(normalized_intensity, dim=2)
            corrected_image = raw_image * avg_intensity
        
        return corrected_image
    
    def get_interference_pattern(self, channel: str = 'combined') -> torch.Tensor:
        """Получение интерференционной картины для визуализации"""
        if not self.is_normalized:
            return torch.zeros((self.height, self.width))
        
        if channel == 'combined':
            # Комбинированная интерференция
            combined = (
                torch.abs(self.rgb_interference_maps['red']) +
                torch.abs(self.rgb_interference_maps['green']) +
                torch.abs(self.rgb_interference_maps['blue'])
            ) / 3.0
            return combined
        elif channel in self.rgb_interference_maps:
            return torch.abs(self.rgb_interference_maps[channel])
        else:
            return torch.zeros((self.height, self.width))
    
    def reset(self):
        """Сброс накопленных данных для новой нормализации"""
        self.amplitude_map = torch.zeros_like(self.amplitude_map)
        self.frequency_map = torch.zeros_like(self.frequency_map) 
        self.phase_map = torch.zeros_like(self.phase_map)
        self.frame_count = 0
        
        for channel in self.rgb_interference_maps:
            self.rgb_interference_maps[channel] = torch.zeros_like(self.rgb_interference_maps[channel])
        
        self.normalized_intensity = torch.zeros_like(self.normalized_intensity)
        self.is_normalized = False
        print("🔄 Карта интерференции сброшена")


class RenderingBackendInterface(ABC):
    """Абстрактный интерфейс для backend'ов рендеринга"""
    
    def __init__(self, config: PathIntegralConfig):
        self.config = config
        self.device = self._setup_device()
        
    @abstractmethod
    def _setup_device(self):
        """Настройка устройства для вычислений"""
        pass
        
    @abstractmethod
    def render_frame(self, scene_sdf, camera_data) -> Tuple[np.ndarray, float, Optional[Dict[str, torch.Tensor]]]:
        """Рендеринг одного кадра
        
        Возвращает:
        - img_array: изображение
        - fps: кадры в секунду  
        - raw_amplitudes: словарь сырых амплитуд по каналам (для нормализации)
        """
        pass
        
    @abstractmethod
    def normalize_vector(self, v):
        """Нормализация вектора"""
        pass
        
    @abstractmethod
    def scene_distance(self, points, scene_sdf):
        """Вычисление расстояний до сцены"""
        pass


class GPUMPSBackend(RenderingBackendInterface):
    """Backend для Apple Silicon GPU (Metal Performance Shaders)"""
    
    def _setup_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
            
    def render_frame(self, scene_sdf, camera_data) -> Tuple[np.ndarray, float, Optional[Dict[str, torch.Tensor]]]:
        """Основной метод рендеринга для MPS"""
        start_time = time.time()
        
        raw_amplitudes = None
        
        if self.config.rendering_mode == RenderingMode.RGB_SPECTRAL:
            img_array, raw_amplitudes = self._render_spectral(scene_sdf, camera_data)
        else:
            img_array, raw_amplitudes = self._render_monochrome(scene_sdf, camera_data)
            
        render_time = time.time() - start_time
        fps = 1.0 / render_time if render_time > 0 else 0
        
        return img_array, fps, raw_amplitudes
        
    def _render_spectral(self, scene_sdf, camera_data):
        """Спектральный RGB рендеринг"""
        # Создаем RGB буферы
        rgb_buffer = torch.zeros(self.config.height, self.config.width, 3, device=self.device)
        raw_amplitudes = {}
        
        # Рендерим каждый канал отдельно
        for i, (color, wavelength) in enumerate(self.config.wavelengths.items()):
            if color in ['red', 'green', 'blue']:
                k = 2.0 * math.pi / wavelength
                channel_intensity, channel_amplitude = self._render_channel(scene_sdf, camera_data, k)
                rgb_buffer[:, :, i] = channel_intensity
                
                # Сохраняем сырые амплитуды для нормализации
                raw_amplitudes[color] = channel_amplitude
                
        # Пост-обработка
        rgb_buffer = torch.clamp(rgb_buffer, 0.0, 1.0)
        gamma_corrected = torch.pow(rgb_buffer, 1.0/2.2)
        return (gamma_corrected * 255.0).byte().cpu().numpy(), raw_amplitudes
        
    def _render_monochrome(self, scene_sdf, camera_data):
        """Монохромный рендеринг"""
        k = 2.0 * math.pi / self.config.wavelengths['monochrome']
        intensity, amplitude = self._render_channel(scene_sdf, camera_data, k)
        
        intensity = torch.clamp(intensity, 0.0, 1.0)
        gamma_corrected = torch.pow(intensity, 1.0/2.2)
        
        # Для монохромного режима возвращаем амплитуду в формате совместимом с RGB
        raw_amplitudes = {'monochrome': amplitude}
        
        return (gamma_corrected * 255.0).byte().cpu().numpy(), raw_amplitudes
        
    def _render_channel(self, scene_sdf, camera_data, k_value):
        """Рендеринг одного спектрального канала"""
        # Получаем направления лучей
        ray_dirs = self._compute_ray_directions(camera_data)
        
        # Инициализация буферов
        intensity_buffer = torch.zeros(self.config.height, self.config.width, device=self.device)
        amplitude_buffer = torch.zeros(self.config.height, self.config.width, device=self.device, dtype=torch.complex64)
        
        # Батчевый рендеринг для оптимизации памяти
        batch_size = min(8, self.config.spp)
        num_batches = (self.config.spp + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, self.config.spp)
            current_batch_size = batch_end - batch_start
            
            # Генерация случайных путей
            path_positions = self._generate_paths(ray_dirs, camera_data['position'], current_batch_size)
            
            # Трассировка путей
            batch_intensities, batch_amplitudes = self._trace_paths(path_positions, scene_sdf, k_value)
            intensity_buffer += batch_intensities.sum(dim=0)
            amplitude_buffer += batch_amplitudes.sum(dim=0)
            
        final_intensity = intensity_buffer / self.config.spp
        final_amplitude = amplitude_buffer / self.config.spp
        
        return final_intensity, final_amplitude
        
    def _compute_ray_directions(self, camera_data):
        """Вычисление направлений лучей для всех пикселей"""
        # Создаем сетку пикселей
        aspect = self.config.width / self.config.height
        screen_h = 2.0 * math.tan(math.radians(self.config.fov) / 2.0)
        screen_w = screen_h * aspect
        
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.config.height, device=self.device, dtype=torch.float32),
            torch.arange(self.config.width, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Нормализованные координаты
        u = (x_coords + 0.5) / self.config.width - 0.5
        v = (y_coords + 0.5) / self.config.height - 0.5
        screen_u = u * screen_w
        screen_v = v * screen_h
        
        # Направления лучей
        cam_pos = camera_data['position']
        cam_dir = camera_data['direction']
        cam_right = camera_data['right']
        cam_up = camera_data['up']
        
        pixel_centers = (
            cam_pos + cam_dir +
            cam_right * screen_u.unsqueeze(-1) +
            cam_up * screen_v.unsqueeze(-1)
        )
        
        return self.normalize_vector(pixel_centers - cam_pos)
        
    def _generate_paths(self, ray_dirs, cam_pos, batch_size):
        """Генерация случайных путей для Path Integral"""
        # Параметры пути
        t_vals = torch.linspace(0.5, 8.0, self.config.segments, device=self.device)
        
        # Базовые позиции
        base_positions = (
            cam_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0) +
            ray_dirs.unsqueeze(0).unsqueeze(3) * 
            t_vals.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        )
        
        # Случайные смещения
        path_offsets = torch.normal(
            0, self.config.jitter_scale,
            size=(batch_size, self.config.height, self.config.width, self.config.segments, 3),
            device=self.device
        )
        
        return base_positions + path_offsets
        
    def _trace_paths(self, path_positions, scene_sdf, k_value):
        """Трассировка путей и вычисление интенсивности с амплитудами"""
        batch_size = path_positions.shape[0]
        batch_intensities = torch.zeros(batch_size, self.config.height, self.config.width, device=self.device)
        batch_amplitudes = torch.zeros(batch_size, self.config.height, self.config.width, device=self.device, dtype=torch.complex64)
        
        for i in range(batch_size):
            positions = path_positions[i]
            
            # Проверка пересечений и накопление оптической длины
            min_distances = torch.full((self.config.height, self.config.width), float('inf'), device=self.device)
            optical_lengths = torch.zeros(self.config.height, self.config.width, device=self.device)
            
            for seg_idx in range(self.config.segments):
                pos = positions[:, :, seg_idx, :]
                pos_flat = pos.reshape(-1, 3)
                
                # SDF проверка
                distances = scene_sdf(pos_flat).reshape(self.config.height, self.config.width)
                min_distances = torch.minimum(min_distances, distances)
                
                # Накопление длины
                if seg_idx > 0:
                    prev_pos = positions[:, :, seg_idx-1, :]
                    segment_length = torch.norm(pos - prev_pos, dim=-1)
                    optical_lengths += segment_length
                    
            # Вычисление квантовой амплитуды
            hit_mask = min_distances < self.config.hit_eps
            phases = k_value * optical_lengths
            
            # Дисперсионные эффекты
            wavelength = 2.0 * math.pi / k_value
            dispersion_factor = 1.0 + 0.3 * (0.55 / wavelength - 1.0)
            
            # Затухание
            attenuation = 1.0 / (1.0 + 0.05 * optical_lengths * dispersion_factor)
            
            # Комплексная амплитуда с учетом фазы
            complex_amplitude = attenuation.unsqueeze(-1) * torch.stack([
                torch.cos(phases), torch.sin(phases)
            ], dim=-1)
            
            # Преобразуем в torch.complex64
            complex_amplitude = torch.complex(complex_amplitude[..., 0], complex_amplitude[..., 1])
            
            # Фоновая составляющая для промахов
            background_amplitude = torch.tensor(0.01, device=self.device, dtype=torch.complex64)
            
            amplitude = torch.where(
                hit_mask,
                complex_amplitude,
                background_amplitude
            )
            
            # Интенсивность = |амплитуда|²
            intensity = torch.abs(amplitude) ** 2
            
            batch_intensities[i] = intensity
            batch_amplitudes[i] = amplitude
            
        return batch_intensities, batch_amplitudes
        
    def normalize_vector(self, v):
        """Нормализация вектора на GPU"""
        return torch.nn.functional.normalize(v, dim=-1)
        
    def scene_distance(self, points, scene_sdf):
        """Вычисление расстояний до сцены"""
        return scene_sdf(points)


class CPUNumpyBackend(RenderingBackendInterface):
    """Простой CPU backend на NumPy (для совместимости)"""
    
    def _setup_device(self):
        return "cpu"
        
    def render_frame(self, scene_sdf, camera_data) -> Tuple[np.ndarray, float, Optional[Dict[str, torch.Tensor]]]:
        """Упрощенный CPU рендеринг"""
        start_time = time.time()
        
        # Упрощенная реализация для демонстрации
        img = np.random.rand(self.config.height, self.config.width) * 128
        img = img.astype(np.uint8)
        
        # Заглушка для raw_amplitudes
        raw_amplitudes = None
        
        render_time = time.time() - start_time
        fps = 1.0 / render_time if render_time > 0 else 0
        
        return img, fps, raw_amplitudes
        
    def normalize_vector(self, v):
        """Нормализация вектора на CPU"""
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return np.where(norm > 1e-8, v / norm, v)
        
    def scene_distance(self, points, scene_sdf):
        """Вычисление расстояний до сцены на CPU"""
        return scene_sdf(points)


class PathIntegralEngine:
    """Главный класс движка рендеринга"""
    
    def __init__(self, config: Optional[PathIntegralConfig] = None):
        self.config = config or PathIntegralConfig()
        self.backend = self._create_backend()
        self.stats = {
            'frames_rendered': 0,
            'total_time': 0.0,
            'avg_fps': 0.0
        }
        
        # Система нормализации частот фотонов
        self.frequency_map = None
        if self.config.enable_normalization:
            self.frequency_map = PhotonFrequencyMap(
                self.config.width, 
                self.config.height, 
                self.config.normalization_frames,
                device=self.backend.device
            )
            print(f"🔬 Система нормализации включена ({self.config.normalization_frames} кадров)")
        
    def _create_backend(self) -> RenderingBackendInterface:
        """Создание backend'а в зависимости от конфигурации"""
        if self.config.backend == RenderingBackend.GPU_MPS:
            return GPUMPSBackend(self.config)
        elif self.config.backend == RenderingBackend.CPU_NUMPY:
            return CPUNumpyBackend(self.config)
        else:
            # Fallback на CPU
            return CPUNumpyBackend(self.config)
            
    def render(self, scene_sdf_func, camera_position, camera_target) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Основной метод рендеринга с поддержкой нормализации частот"""
        # Подготовка данных камеры
        camera_data = self._prepare_camera_data(camera_position, camera_target)
        
        # Рендеринг кадра
        img_array, fps, raw_amplitudes = self.backend.render_frame(scene_sdf_func, camera_data)
        
        # Система нормализации частот фотонов
        if self.frequency_map is not None:
            # Накапливаем данные для нормализации
            if raw_amplitudes and not self.frequency_map.is_normalized:
                self.frequency_map.accumulate_frame(raw_amplitudes, self.config.wavelengths)
            
            # Применяем нормализацию если она готова
            if self.frequency_map.is_normalized:
                if isinstance(img_array, np.ndarray):
                    img_tensor = torch.from_numpy(img_array)
                else:
                    img_tensor = img_array
                
                # Применяем интерференционную коррекцию
                normalized_tensor = self.frequency_map.apply_normalization(img_tensor)
                
                # Смешиваем с исходным изображением по силе нормализации
                blend_factor = self.config.normalization_strength
                img_tensor = (1.0 - blend_factor) * img_tensor + blend_factor * normalized_tensor
                
                img_array = img_tensor.cpu().numpy() if isinstance(img_tensor, torch.Tensor) else img_tensor
        
        # Обновление статистики
        self._update_stats(fps)
        
        # Адаптивное управление качеством
        if self.config.adaptive_quality:
            self._adjust_quality(fps)
            
        info = {
            'fps': fps,
            'spp': self.config.spp,
            'backend': self.config.backend.value,
            'mode': self.config.rendering_mode.value,
            'stats': self.stats.copy()
        }
        
        # Добавляем информацию о нормализации
        if self.frequency_map is not None:
            info['normalization'] = {
                'enabled': True,
                'frames_accumulated': self.frequency_map.frame_count,
                'total_frames': self.frequency_map.normalization_frames,
                'is_ready': self.frequency_map.is_normalized,
                'strength': self.config.normalization_strength
            }
        else:
            info['normalization'] = {'enabled': False}
            
        return img_array, info
        
    def _prepare_camera_data(self, position, target):
        """Подготовка данных камеры"""
        if isinstance(position, (list, tuple)):
            position = torch.tensor(position, device=self.backend.device, dtype=torch.float32)
        if isinstance(target, (list, tuple)):
            target = torch.tensor(target, device=self.backend.device, dtype=torch.float32)
            
        # Вычисление базиса камеры
        cam_dir = self.backend.normalize_vector(target - position)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.backend.device)
        
        cam_right = self.backend.normalize_vector(torch.linalg.cross(cam_dir, world_up))
        cam_up = torch.linalg.cross(cam_right, cam_dir)
        
        return {
            'position': position,
            'direction': cam_dir,
            'right': cam_right,
            'up': cam_up,
            'target': target
        }
        
    def _update_stats(self, fps):
        """Обновление статистики рендеринга"""
        self.stats['frames_rendered'] += 1
        self.stats['total_time'] += 1.0 / fps if fps > 0 else 0
        
        if self.stats['frames_rendered'] > 0:
            self.stats['avg_fps'] = self.stats['frames_rendered'] / self.stats['total_time']
            
    def _adjust_quality(self, current_fps):
        """Адаптивное управление качеством"""
        if current_fps < self.config.target_fps * 0.8:
            # Снижаем качество
            self.config.spp = max(self.config.min_spp, self.config.spp - 2)
        elif current_fps > self.config.target_fps * 1.2:
            # Повышаем качество
            self.config.spp = min(self.config.max_spp, self.config.spp + 1)
            
    def set_quality_preset(self, preset: QualityPreset):
        """Установка предустановки качества"""
        new_config = PathIntegralConfig.from_preset(preset)
        
        # Сохраняем backend и режим рендеринга
        new_config.backend = self.config.backend
        new_config.rendering_mode = self.config.rendering_mode
        
        self.config = new_config
        self.backend = self._create_backend()
        
    def switch_rendering_mode(self, mode: RenderingMode):
        """Переключение режима рендеринга"""
        self.config.rendering_mode = mode
        
    def get_info(self) -> Dict[str, Any]:
        """Получение информации о движке"""
        info = {
            'version': '3.0',
            'backend': self.config.backend.value,
            'device': str(self.backend.device),
            'rendering_mode': self.config.rendering_mode.value,
            'resolution': f"{self.config.width}x{self.config.height}",
            'spp': self.config.spp,
            'stats': self.stats
        }
        
        # Добавляем информацию о нормализации
        if self.frequency_map is not None:
            info['normalization'] = {
                'enabled': True,
                'frames_accumulated': self.frequency_map.frame_count,
                'total_frames': self.frequency_map.normalization_frames,
                'is_ready': self.frequency_map.is_normalized,
                'strength': self.config.normalization_strength
            }
        else:
            info['normalization'] = {'enabled': False}
            
        return info
    
    def enable_normalization(self, frames: int = 100, strength: float = 1.0):
        """Включение системы нормализации частот фотонов"""
        self.config.enable_normalization = True
        self.config.normalization_frames = frames
        self.config.normalization_strength = max(0.0, min(1.0, strength))
        
        # Создаем новую карту если её нет или изменились параметры
        if (self.frequency_map is None or 
            self.frequency_map.normalization_frames != frames):
            self.frequency_map = PhotonFrequencyMap(
                self.config.width, 
                self.config.height, 
                frames,
                device=self.backend.device
            )
        
        print(f"✅ Нормализация включена: {frames} кадров, сила {strength:.1f}")
    
    def disable_normalization(self):
        """Отключение системы нормализации"""
        self.config.enable_normalization = False
        self.frequency_map = None
        print("❌ Нормализация отключена")
    
    def reset_normalization(self):
        """Сброс накопленных данных нормализации"""
        if self.frequency_map is not None:
            self.frequency_map.reset()
            print("🔄 Данные нормализации сброшены")
    
    def set_normalization_strength(self, strength: float):
        """Установка силы применения нормализации (0.0-1.0)"""
        self.config.normalization_strength = max(0.0, min(1.0, strength))
        print(f"⚙️  Сила нормализации: {self.config.normalization_strength:.1f}")
    
    def get_interference_pattern(self, channel: str = 'combined') -> Optional[np.ndarray]:
        """Получение интерференционной картины для визуализации"""
        if self.frequency_map is not None and self.frequency_map.is_normalized:
            pattern = self.frequency_map.get_interference_pattern(channel)
            return pattern.cpu().numpy() if isinstance(pattern, torch.Tensor) else pattern
        return None
    
    def export_normalization_data(self, filepath: str):
        """Экспорт данных нормализации в файл"""
        if self.frequency_map is not None and self.frequency_map.is_normalized:
            data = {
                'width': self.frequency_map.width,
                'height': self.frequency_map.height,
                'frames': self.frequency_map.normalization_frames,
                'normalized_intensity': self.frequency_map.normalized_intensity.cpu().numpy(),
                'rgb_interference': {
                    channel: torch.abs(complex_map).cpu().numpy()
                    for channel, complex_map in self.frequency_map.rgb_interference_maps.items()
                }
            }
            
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Данные нормализации сохранены: {filepath}")
        else:
            print("⚠️  Нет данных для экспорта")


# Заводские функции для удобства
def create_interactive_engine() -> PathIntegralEngine:
    """Создание движка для интерактивного использования"""
    config = PathIntegralConfig.from_preset(QualityPreset.INTERACTIVE)
    return PathIntegralEngine(config)


def create_production_engine() -> PathIntegralEngine:
    """Создание движка для высококачественного рендеринга"""
    config = PathIntegralConfig.from_preset(QualityPreset.PRODUCTION)
    return PathIntegralEngine(config)


def create_preview_engine() -> PathIntegralEngine:
    """Создание движка для быстрого предпросмотра"""
    config = PathIntegralConfig.from_preset(QualityPreset.PREVIEW)
    return PathIntegralEngine(config)


# Пример использования
if __name__ == "__main__":
    print("Path Integral Rendering Engine v3.0")
    print("====================================")
    
    # Создание движка
    engine = create_interactive_engine()
    print(f"Движок создан: {engine.get_info()}")
    
    # Простая SDF сцена для демонстрации
    def simple_scene_sdf(points):
        """Простая сцена с тремя сферами"""
        if hasattr(points, 'device'):  # PyTorch tensor
            device = points.device
            # Сфера в центре
            d1 = torch.norm(points - torch.tensor([0.0, 0.0, 0.0], device=device), dim=-1) - 1.0
            # Сфера справа
            d2 = torch.norm(points - torch.tensor([2.0, 0.5, 1.0], device=device), dim=-1) - 0.7
            # Плоскость-пол
            d3 = points[..., 1] + 1.5
            return torch.minimum(torch.minimum(d1, d2), d3)
        else:  # NumPy array
            # Fallback для CPU
            return np.ones(points.shape[:-1]) * 0.5
    
    # Тестовый рендеринг
    try:
        img, info = engine.render(
            scene_sdf_func=simple_scene_sdf,
            camera_position=[0.0, 1.0, -3.0],
            camera_target=[0.0, 0.0, 0.0]
        )
        print(f"Кадр отрендерен: {img.shape}, FPS: {info['fps']:.2f}")
    except Exception as e:
        print(f"Ошибка рендеринга: {e}")
