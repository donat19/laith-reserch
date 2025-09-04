"""
Демонстрация Path Integral Engine v3.0
======================================
Примеры использования унифицированного движка рендеринга

Запуск: python engine_demo.py
"""

import sys
import time
import numpy as np
from pathlib import Path

# Добавляем текущую директорию в path для импорта движка
sys.path.append(str(Path(__file__).parent))

try:
    from path_integral_engine import (
        PathIntegralEngine, 
        PathIntegralConfig,
        QualityPreset,
        RenderingMode,
        RenderingBackend,
        create_interactive_engine,
        create_production_engine,
        create_preview_engine
    )
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Не удалось импортировать зависимости: {e}")
    print("Установите зависимости: pip install torch numpy")
    TORCH_AVAILABLE = False


def demo_scene_sdf(points):
    """Демонстрационная сцена с несколькими объектами"""
    if not hasattr(points, 'device'):
        # Fallback для CPU
        return np.ones(points.shape[:-1]) * 0.5
        
    device = points.device
    
    # Центральная сфера
    sphere1 = torch.norm(points - torch.tensor([0.0, 0.0, 0.0], device=device), dim=-1) - 1.0
    
    # Сфера справа
    sphere2 = torch.norm(points - torch.tensor([2.5, 0.0, 0.5], device=device), dim=-1) - 0.8
    
    # Сфера слева
    sphere3 = torch.norm(points - torch.tensor([-2.0, 0.5, -0.5], device=device), dim=-1) - 0.6
    
    # Сфера сверху
    sphere4 = torch.norm(points - torch.tensor([0.0, 2.0, 0.0], device=device), dim=-1) - 0.7
    
    # Плоскость-пол
    plane = points[..., 1] + 1.8
    
    # Объединяем все объекты
    scene = torch.minimum(
        torch.minimum(
            torch.minimum(
                torch.minimum(sphere1, sphere2), 
                sphere3
            ), 
            sphere4
        ), 
        plane
    )
    
    return scene


def save_image(img_array, filename, info):
    """Сохранение изображения"""
    try:
        from PIL import Image
        
        if len(img_array.shape) == 3:  # RGB
            img = Image.fromarray(img_array)
        else:  # Grayscale
            img = Image.fromarray(img_array, mode='L')
            
        img.save(filename)
        print(f"✅ Изображение сохранено: {filename}")
        print(f"   Размер: {img_array.shape}")
        print(f"   FPS: {info['fps']:.2f}")
        print(f"   SPP: {info['spp']}")
        print(f"   Backend: {info['backend']}")
        print(f"   Режим: {info['mode']}")
        print()
        
    except ImportError:
        print("⚠️  PIL не найден, изображение не сохранено")
        print(f"Массив изображения: {img_array.shape}, FPS: {info['fps']:.2f}")


def demo_basic_usage():
    """Базовая демонстрация использования движка"""
    print("🎬 Демонстрация базового использования")
    print("=" * 40)
    
    # Создание движка для интерактивного использования
    engine = create_interactive_engine()
    print(f"Движок создан: {engine.get_info()}")
    
    # Позиции камеры
    camera_positions = [
        ([0.0, 1.0, -4.0], [0.0, 0.0, 0.0], "front"),
        ([4.0, 2.0, 0.0], [0.0, 0.0, 0.0], "side"),
        ([2.0, 3.0, 2.0], [0.0, 0.0, 0.0], "angle"),
    ]
    
    for i, (cam_pos, cam_target, view_name) in enumerate(camera_positions):
        print(f"🎥 Рендеринг вида '{view_name}'...")
        
        start_time = time.time()
        img, info = engine.render(demo_scene_sdf, cam_pos, cam_target)
        render_time = time.time() - start_time
        
        filename = f"demo_basic_{view_name}.png"
        save_image(img, filename, info)
        print(f"   Время рендера: {render_time:.2f}с\n")


def demo_quality_presets():
    """Демонстрация различных предустановок качества"""
    print("🎯 Демонстрация предустановок качества")
    print("=" * 40)
    
    presets = [
        (QualityPreset.PREVIEW, "preview"),
        (QualityPreset.INTERACTIVE, "interactive"),
        (QualityPreset.PRODUCTION, "production"),
    ]
    
    camera_pos = [3.0, 2.0, -3.0]
    camera_target = [0.0, 0.0, 0.0]
    
    for preset, name in presets:
        print(f"🎨 Рендеринг с предустановкой '{name}'...")
        
        # Создание конфигурации
        config = PathIntegralConfig.from_preset(preset)
        engine = PathIntegralEngine(config)
        
        start_time = time.time()
        img, info = engine.render(demo_scene_sdf, camera_pos, camera_target)
        render_time = time.time() - start_time
        
        filename = f"demo_quality_{name}.png"
        save_image(img, filename, info)
        print(f"   Время рендера: {render_time:.2f}с")
        print(f"   Разрешение: {config.width}x{config.height}")
        print(f"   SPP: {config.spp}\n")


def demo_rendering_modes():
    """Демонстрация различных режимов рендеринга"""
    print("🌈 Демонстрация режимов рендеринга")
    print("=" * 40)
    
    engine = create_interactive_engine()
    camera_pos = [2.0, 1.5, -3.0]
    camera_target = [0.0, 0.0, 0.0]
    
    # RGB спектральный режим
    print("🔴🟢🔵 RGB Спектральный режим...")
    engine.switch_rendering_mode(RenderingMode.RGB_SPECTRAL)
    img_rgb, info_rgb = engine.render(demo_scene_sdf, camera_pos, camera_target)
    save_image(img_rgb, "demo_mode_rgb_spectral.png", info_rgb)
    
    # Монохромный режим  
    print("⚫ Монохромный режим...")
    engine.switch_rendering_mode(RenderingMode.MONOCHROME)
    img_mono, info_mono = engine.render(demo_scene_sdf, camera_pos, camera_target)
    save_image(img_mono, "demo_mode_monochrome.png", info_mono)


def demo_custom_config():
    """Демонстрация пользовательской конфигурации"""
    print("⚙️  Демонстрация пользовательской конфигурации")
    print("=" * 40)
    
    # Создание пользовательской конфигурации
    config = PathIntegralConfig()
    config.width = 768
    config.height = 768
    config.spp = 32
    config.segments = 6
    config.rendering_mode = RenderingMode.RGB_SPECTRAL
    config.target_fps = 5.0  # Более высокое качество
    config.adaptive_quality = False  # Фиксированное качество
    
    # Кастомные длины волн (более насыщенные цвета)
    config.wavelengths['red'] = 0.680    # Более красный
    config.wavelengths['green'] = 0.520  # Более зеленый
    config.wavelengths['blue'] = 0.430   # Более синий
    
    print(f"Пользовательская конфигурация:")
    print(f"  Разрешение: {config.width}x{config.height}")
    print(f"  SPP: {config.spp}")
    print(f"  Сегменты: {config.segments}")
    print(f"  RGB длины волн: R={config.wavelengths['red']}μm, "
          f"G={config.wavelengths['green']}μm, B={config.wavelengths['blue']}μm")
    
    engine = PathIntegralEngine(config)
    
    camera_pos = [3.0, 2.5, -2.5]
    camera_target = [0.0, 0.0, 0.0]
    
    print("\n🎨 Рендеринг с пользовательскими настройками...")
    start_time = time.time()
    img, info = engine.render(demo_scene_sdf, camera_pos, camera_target)
    render_time = time.time() - start_time
    
    save_image(img, "demo_custom_config.png", info)
    print(f"Время рендера: {render_time:.2f}с")


def demo_performance_comparison():
    """Сравнение производительности разных настроек"""
    print("📊 Сравнение производительности")
    print("=" * 40)
    
    camera_pos = [2.0, 1.0, -3.0]
    camera_target = [0.0, 0.0, 0.0]
    
    test_configs = [
        ("Быстрый", QualityPreset.PREVIEW),
        ("Средний", QualityPreset.INTERACTIVE),
        ("Медленный", QualityPreset.PRODUCTION),
    ]
    
    results = []
    
    for name, preset in test_configs:
        config = PathIntegralConfig.from_preset(preset)
        engine = PathIntegralEngine(config)
        
        print(f"⏱️  Тест '{name}' ({config.width}x{config.height}, {config.spp} SPP)...")
        
        # Делаем несколько прогонов для точности
        times = []
        for i in range(3):
            start_time = time.time()
            img, info = engine.render(demo_scene_sdf, camera_pos, camera_target)
            render_time = time.time() - start_time
            times.append(render_time)
            
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        results.append((name, config.width * config.height, config.spp, avg_time, fps))
        print(f"   Среднее время: {avg_time:.2f}с, FPS: {fps:.2f}")
    
    print("\n📈 Сводка производительности:")
    print(f"{'Режим':<10} {'Пиксели':<10} {'SPP':<6} {'Время':<8} {'FPS':<8}")
    print("-" * 50)
    for name, pixels, spp, time_val, fps in results:
        print(f"{name:<10} {pixels:<10} {spp:<6} {time_val:<8.2f} {fps:<8.2f}")


def main():
    """Главная функция демонстрации"""
    print("🚀 Path Integral Engine v3.0 - Демонстрация")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("❌ Демонстрация недоступна без PyTorch")
        return
        
    # Проверка доступности GPU
    if torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU (MPS) доступен")
    else:
        print("⚠️  MPS недоступен, используется CPU")
    
    print()
    
    try:
        # Запуск всех демонстраций
        demo_basic_usage()
        demo_quality_presets()
        demo_rendering_modes()
        demo_custom_config()
        demo_performance_comparison()
        
        print("🎉 Все демонстрации завершены!")
        print("📁 Проверьте сгенерированные изображения:")
        print("   - demo_basic_*.png")
        print("   - demo_quality_*.png") 
        print("   - demo_mode_*.png")
        print("   - demo_custom_config.png")
        
    except Exception as e:
        print(f"❌ Ошибка во время демонстрации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
