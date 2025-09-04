"""
Демонстрация Path Integral Engine v3.0
======================================
Примеры использования унифицированного движка рендеринга

Запуск: python engine_demo.py
"""

import sys
import time
import math
import numpy as np
import os
from pathlib import Path

# Добавляем текущую директорию в path для импорта движка
sys.path.append(str(Path(__file__).    # Экспорт данных нормализации
    normalization_data_path = os.path.join(output_dir, "normalization_data.pkl")
    engine.export_normalization_data(normalization_data_path)
    
    # Создаем финальную композитную картинку
    print("🎨 Создание финальной композитной картинки...")
    create_final_composite_image(engine, demo_scene_sdf, camera_pos, camera_target, output_dir)
    
    print("\n📊 Информация о нормализации:")
    norm_info = engine.get_info().get('normalization', {})
    print(f"   Кадров накоплено: {norm_info.get('frames_accumulated', 0)}")
    print(f"   Всего кадров: {norm_info.get('total_frames', 0)}")
    print(f"   Готовность: {'Да' if norm_info.get('is_ready', False) else 'Нет'}")
    print(f"   Сила применения: {norm_info.get('strength', 0.0):.1f}")


def create_final_composite_image(engine, scene_sdf, camera_pos, camera_target, output_dir):
    """Создание финальной композитной картинки с градиентом нормализации"""
    
    # Сохраняем исходную силу нормализации
    original_strength = engine.config.normalization_strength
    
    # Создаем несколько изображений с разной силой нормализации
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    images = []
    
    print("   Рендеринг градиента нормализации...")
    for i, strength in enumerate(strengths):
        print(f"     Сила {strength:.2f} ({i+1}/{len(strengths)})")
        engine.set_normalization_strength(strength)
        img, info = engine.render(scene_sdf, camera_pos, camera_target)
        
        # Конвертируем в правильный формат
        if len(img.shape) == 3 and img.dtype == np.uint8:
            img_normalized = img.astype(np.float32) / 255.0
        else:
            img_normalized = img
            
        images.append(img_normalized)
    
    # Восстанавливаем исходную силу
    engine.set_normalization_strength(original_strength)
    
    # Получаем карты интерференции
    interference_combined = engine.get_interference_pattern('combined')
    interference_red = engine.get_interference_pattern('red')
    interference_green = engine.get_interference_pattern('green')
    interference_blue = engine.get_interference_pattern('blue')
    
    # Создаем композитную картинку
    try:
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # Размеры для композита
        img_height, img_width = images[0].shape[:2]
        
        # Создаем большую композитную картинку
        composite_width = img_width * 5  # 5 изображений по горизонтали
        composite_height = img_height * 3  # 3 ряда
        
        composite = np.ones((composite_height, composite_width, 3), dtype=np.float32)
        
        # Первый ряд: градиент нормализации
        for i, img in enumerate(images):
            x_start = i * img_width
            x_end = x_start + img_width
            
            if len(img.shape) == 3:
                composite[0:img_height, x_start:x_end, :] = img
            else:
                # Монохромное изображение - дублируем на все каналы
                composite[0:img_height, x_start:x_end, :] = np.stack([img] * 3, axis=2)
        
        # Второй ряд: карты интерференции
        interference_maps = [interference_red, interference_green, interference_blue, interference_combined]
        interference_names = ['Red', 'Green', 'Blue', 'Combined']
        
        for i, (interference_map, name) in enumerate(zip(interference_maps, interference_names)):
            if interference_map is not None:
                x_start = i * img_width
                x_end = x_start + img_width
                y_start = img_height
                y_end = y_start + img_height
                
                # Нормализуем карту интерференции
                map_normalized = (interference_map - interference_map.min()) / (interference_map.max() - interference_map.min() + 1e-8)
                
                # Применяем цветовую карту для визуализации
                if i < 3:  # RGB каналы
                    colored_map = np.zeros((img_height, img_width, 3))
                    colored_map[:, :, i] = map_normalized  # Отображаем в соответствующий цветовой канал
                else:  # Combined
                    colored_map = plt.cm.viridis(map_normalized)[:, :, :3]  # Используем цветовую карту viridis
                
                composite[y_start:y_end, x_start:x_end, :] = colored_map
        
        # Третий ряд: сравнение до/после и анализ
        # Изображение без нормализации
        img_before = images[0]  # strength 0.0
        img_after = images[-1]  # strength 1.0
        
        # Разностное изображение
        diff_img = np.abs(img_after - img_before)
        diff_img = diff_img / (diff_img.max() + 1e-8)  # Нормализуем
        
        # Градиентная карта силы
        gradient_map = np.zeros((img_height, img_width, 3))
        for i in range(img_width):
            strength_val = i / img_width
            gradient_map[:, i, :] = plt.cm.plasma(strength_val)[:3]  # Градиент от 0 до 1
        
        # Заполняем третий ряд
        y_start = img_height * 2
        y_end = y_start + img_height
        
        # До нормализации
        if len(img_before.shape) == 3:
            composite[y_start:y_end, 0:img_width, :] = img_before
        else:
            composite[y_start:y_end, 0:img_width, :] = np.stack([img_before] * 3, axis=2)
        
        # После нормализации
        if len(img_after.shape) == 3:
            composite[y_start:y_end, img_width:img_width*2, :] = img_after
        else:
            composite[y_start:y_end, img_width:img_width*2, :] = np.stack([img_after] * 3, axis=2)
        
        # Разность
        composite[y_start:y_end, img_width*2:img_width*3, :] = diff_img
        
        # Градиентная карта
        composite[y_start:y_end, img_width*3:img_width*4, :] = gradient_map
        
        # Оставляем последний участок для текста или пустым
        
        # Конвертируем в uint8 и создаем PIL изображение
        composite_uint8 = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
        composite_img = Image.fromarray(composite_uint8)
        
        # Добавляем текстовые метки
        draw = ImageDraw.Draw(composite_img)
        
        # Пытаемся загрузить шрифт, если не получается - используем стандартный
        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 40)
            font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 30)
        except:
            try:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
            except:
                font_large = None
                font_small = None
        
        # Заголовки рядов
        draw.text((10, 10), "Градиент нормализации (сила: 0.0 → 1.0)", fill=(255, 255, 255), font=font_large)
        draw.text((10, img_height + 10), "Карты интерференции по каналам", fill=(255, 255, 255), font=font_large)
        draw.text((10, img_height * 2 + 10), "Анализ: До → После → Разность → Градиент", fill=(255, 255, 255), font=font_large)
        
        # Подписи для первого ряда
        for i, strength in enumerate(strengths):
            x_pos = i * img_width + img_width // 2 - 30
            draw.text((x_pos, img_height - 40), f"{strength:.2f}", fill=(255, 255, 0), font=font_small)
        
        # Подписи для второго ряда
        for i, name in enumerate(interference_names):
            x_pos = i * img_width + img_width // 2 - len(name) * 8
            draw.text((x_pos, img_height * 2 - 40), name, fill=(255, 255, 0), font=font_small)
        
        # Подписи для третьего ряда
        labels = ["До", "После", "Разность", "Градиент"]
        for i, label in enumerate(labels):
            x_pos = i * img_width + img_width // 2 - len(label) * 8
            draw.text((x_pos, composite_height - 40), label, fill=(255, 255, 0), font=font_small)
        
        # Сохраняем финальную композитную картинку
        final_path = os.path.join(output_dir, "FINAL_NORMALIZATION_COMPOSITE.png")
        composite_img.save(final_path, quality=95)
        
        print(f"✅ Финальная композитная картинка сохранена: {final_path}")
        print(f"   Размер: {composite_width}x{composite_height}")
        print(f"   Содержимое:")
        print(f"     Ряд 1: Градиент нормализации (5 изображений)")
        print(f"     Ряд 2: Карты интерференции (R, G, B, Combined)")
        print(f"     Ряд 3: Анализ (До/После/Разность/Градиент)")
        
    except ImportError as e:
        print(f"⚠️  Не удалось создать композитную картинку: {e}")
        print("   Установите matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"❌ Ошибка создания композитной картинки: {e}")


def demo_scene_sdf(points)::
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


def save_image(img_array, filename, info, output_dir="renders"):
    """Сохранение изображения"""
    # Создаем папку если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Полный путь к файлу
    full_path = os.path.join(output_dir, filename)
    
    try:
        from PIL import Image
        
        if len(img_array.shape) == 3:  # RGB
            img_rgb = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_rgb)
        else:  # Grayscale
            img_mono = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_mono)
            
        img.save(full_path)
        print(f"✅ Изображение сохранено: {full_path}")
        print(f"   Размер: {img_array.shape}")
        print(f"   FPS: {info['fps']:.2f}")
        print(f"   SPP: {info['spp']}")
        print(f"   Backend: {info['backend']}")
        print(f"   Режим: {info['mode']}")
        print()
        
    except ImportError:
        print("⚠️  PIL не найден, изображение не сохранено")
        print(f"Массив изображения: {img_array.shape}, FPS: {info['fps']:.2f}")
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")


def demo_basic_usage(output_dir="renders"):
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
        save_image(img, filename, info, output_dir)
        print(f"   Время рендера: {render_time:.2f}с\n")


def demo_quality_presets(output_dir="renders"):
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
        save_image(img, filename, info, output_dir)
        print(f"   Время рендера: {render_time:.2f}с")
        print(f"   Разрешение: {config.width}x{config.height}")
        print(f"   SPP: {config.spp}\n")


def demo_rendering_modes(output_dir="renders"):
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
    save_image(img_rgb, "demo_mode_rgb_spectral.png", info_rgb, output_dir)
    
    # Монохромный режим  
    print("⚫ Монохромный режим...")
    engine.switch_rendering_mode(RenderingMode.MONOCHROME)
    img_mono, info_mono = engine.render(demo_scene_sdf, camera_pos, camera_target)
    save_image(img_mono, "demo_mode_monochrome.png", info_mono, output_dir)


def demo_custom_config(output_dir="renders"):
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
    
    save_image(img, "demo_custom_config.png", info, output_dir)
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


def demo_normalization_system(output_dir="renders"):
    """Демонстрация системы нормализации частот фотонов"""
    print("🔬 Демонстрация системы нормализации частот фотонов")
    print("=" * 50)
    
    # Создаем движок с включенной нормализацией
    engine = create_interactive_engine()
    
    # Включаем нормализацию на 20 кадров для быстрой демонстрации
    engine.enable_normalization(frames=20, strength=0.8)
    
    camera_pos = [3.0, 2.0, -3.0]
    camera_target = [0.0, 0.0, 0.0]
    
    print("📊 Фаза накопления данных (20 кадров)...")
    
    # Рендерим кадры для накопления данных нормализации
    for frame_num in range(20):
        print(f"   Кадр {frame_num + 1}/20", end='\r')
        
        # Небольшое изменение камеры для разнообразия
        angle = frame_num * 0.1
        cam_pos = [
            3.0 * math.cos(angle),
            2.0 + 0.5 * math.sin(angle * 2),
            -3.0 * math.sin(angle)
        ]
        
        img, info = engine.render(demo_scene_sdf, cam_pos, camera_target)
        
        # Проверяем готовность нормализации
        if info.get('normalization', {}).get('is_ready', False):
            print(f"\n✅ Карта интерференции готова на кадре {frame_num + 1}!")
            break
    
    print("\n🎨 Рендеринг с применением нормализации...")
    
    # Тестируем разные уровни силы нормализации
    strengths = [0.0, 0.5, 1.0]
    
    for strength in strengths:
        engine.set_normalization_strength(strength)
        
        img, info = engine.render(demo_scene_sdf, camera_pos, camera_target)
        
        strength_name = f"strength_{strength:.1f}".replace(".", "_")
        filename = f"demo_normalization_{strength_name}.png"
        save_image(img, filename, info, output_dir)
        
        print(f"   Сила нормализации {strength:.1f}: {info['fps']:.2f} FPS")
    
    # Сохраняем карту интерференции для визуализации
    print("🌈 Сохранение карт интерференции...")
    
    interference_combined = engine.get_interference_pattern('combined')
    if interference_combined is not None:
        # Нормализуем для сохранения как изображение
        interference_normalized = (interference_combined * 255).astype(np.uint8)
        
        try:
            from PIL import Image
            img_interference = Image.fromarray(interference_normalized)
            interference_path = os.path.join(output_dir, "interference_pattern_combined.png")
            img_interference.save(interference_path)
            print(f"✅ Карта интерференции сохранена: {interference_path}")
        except Exception as e:
            print(f"⚠️  Ошибка сохранения карты интерференции: {e}")
    
    # Сохраняем RGB карты интерференции
    for channel in ['red', 'green', 'blue']:
        pattern = engine.get_interference_pattern(channel)
        if pattern is not None:
            pattern_normalized = (pattern * 255).astype(np.uint8)
            try:
                from PIL import Image
                img_channel = Image.fromarray(pattern_normalized)
                channel_path = os.path.join(output_dir, f"interference_pattern_{channel}.png")
                img_channel.save(channel_path)
                print(f"✅ Карта {channel} сохранена: {channel_path}")
            except Exception as e:
                print(f"⚠️  Ошибка сохранения карты {channel}: {e}")
    
    # Экспорт данных нормализации
    normalization_data_path = os.path.join(output_dir, "normalization_data.pkl")
    engine.export_normalization_data(normalization_data_path)
    
    print("\n📊 Информация о нормализации:")
    norm_info = engine.get_info().get('normalization', {})
    print(f"   Кадров накоплено: {norm_info.get('frames_accumulated', 0)}")
    print(f"   Всего кадров: {norm_info.get('total_frames', 0)}")
    print(f"   Готовность: {'Да' if norm_info.get('is_ready', False) else 'Нет'}")
    print(f"   Сила применения: {norm_info.get('strength', 0.0):.1f}")


def show_menu():
    """Показать интерактивное меню выбора"""
    print("\n" + "="*60)
    print("🎬 Path Integral Engine v3.0 - Демонстрационное меню")
    print("="*60)
    print("Выберите тип рендеринга:")
    print()
    print("1. 📷 Базовое использование (3 вида камеры) ~4сек")
    print("2. 🎯 Сравнение качества (Preview/Interactive/Production) ~60сек")
    print("3. 🌈 Режимы рендеринга (RGB/Монохром) ~2сек")
    print("4. ⚙️  Пользовательская конфигурация (высокое качество) ~15сек")
    print("5. 📊 Тест производительности (без сохранения изображений) ~180сек")
    print("6. � Система нормализации частот фотонов (НОВОЕ!) ~30сек")
    print("7. �🚀 Полная демонстрация (все вышеперечисленное) ~290сек")
    print("0. ❌ Выход")
    print()
    print("💡 Примерное время указано для Apple Silicon M1/M2/M3")
    print("🔬 Вариант 6 - демонстрация интерференции фотонов за 100 кадров")
    print()
    
    while True:
        try:
            choice = input("Ваш выбор (0-7): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6', '7']:
                return choice
            else:
                print("⚠️  Неверный выбор. Введите число от 0 до 7.")
        except KeyboardInterrupt:
            print("\n👋 Выход из программы...")
            return '0'
        except EOFError:
            return '0'


def get_output_directory():
    """Получить папку для сохранения или создать по умолчанию"""
    print("\n📁 Настройка папки для сохранения изображений:")
    print("1. Использовать папку по умолчанию: './renders'")
    print("2. Указать свою папку")
    
    while True:
        try:
            choice = input("Ваш выбор (1-2): ").strip()
            
            if choice == '1':
                output_dir = "renders"
                break
            elif choice == '2':
                custom_dir = input("Введите путь к папке: ").strip()
                if custom_dir:
                    output_dir = custom_dir
                    break
                else:
                    print("⚠️  Пустой путь. Используется папка по умолчанию.")
                    output_dir = "renders"
                    break
            else:
                print("⚠️  Неверный выбор. Введите 1 или 2.")
        except (KeyboardInterrupt, EOFError):
            print("\nИспользуется папка по умолчанию: './renders'")
            output_dir = "renders"
            break
    
    # Создаем папку если не существует
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Папка готова: {os.path.abspath(output_dir)}")
    except Exception as e:
        print(f"⚠️  Не удалось создать папку {output_dir}: {e}")
        print("Используется папка по умолчанию: './renders'")
        output_dir = "renders"
        os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def run_selected_demo(choice, output_dir):
    """Запустить выбранную демонстрацию"""
    if choice == '1':
        demo_basic_usage(output_dir)
    elif choice == '2':
        demo_quality_presets(output_dir)
    elif choice == '3':
        demo_rendering_modes(output_dir)
    elif choice == '4':
        demo_custom_config(output_dir)
    elif choice == '5':
        demo_performance_comparison()
    elif choice == '6':
        demo_normalization_system(output_dir)
    elif choice == '7':
        # Полная демонстрация
        demo_basic_usage(output_dir)
        demo_quality_presets(output_dir)
        demo_rendering_modes(output_dir)
        demo_custom_config(output_dir)
        demo_performance_comparison()
        demo_normalization_system(output_dir)
        
        print("\n🎉 Полная демонстрация завершена!")
        print(f"📁 Все изображения сохранены в: {os.path.abspath(output_dir)}")


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
    
    try:
        # Показываем меню и получаем выбор пользователя
        choice = show_menu()
        
        if choice == '0':
            print("👋 До свидания!")
            return
        
        # Получаем папку для сохранения
        output_dir = get_output_directory()
        
        print(f"\n🎬 Запуск демонстрации...")
        print(f"📁 Результаты будут сохранены в: {os.path.abspath(output_dir)}")
        print()
        
        # Запускаем выбранную демонстрацию
        start_time = time.time()
        run_selected_demo(choice, output_dir)
        total_time = time.time() - start_time
        
        print(f"\n⏱️  Общее время выполнения: {total_time:.2f} секунд")
        print(f"📁 Проверьте сгенерированные изображения в папке: {os.path.abspath(output_dir)}")
        
        # Спрашиваем о повторном запуске
        print("\n" + "="*50)
        while True:
            try:
                repeat = input("🔄 Запустить ещё одну демонстрацию? (y/n): ").strip().lower()
                if repeat in ['y', 'yes', 'д', 'да']:
                    main()  # Рекурсивный вызов для повторного запуска
                    break
                elif repeat in ['n', 'no', 'н', 'нет']:
                    print("👋 До свидания!")
                    break
                else:
                    print("⚠️  Введите 'y' для да или 'n' для нет")
            except (KeyboardInterrupt, EOFError):
                print("\n👋 До свидания!")
                break
        
    except Exception as e:
        print(f"❌ Ошибка во время демонстрации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
