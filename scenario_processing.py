from image_preparation import prepareObjectImage, affineTransform
from PIL import Image, ImageFont
from moviepy import ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.fx import Rotate, FadeOut, FadeIn, CrossFadeOut, CrossFadeIn
import numpy as np
import os
import json

# Функция загрузки данных из JSON-файла
def loadJSON(filename):
    try:
        with open(filename, "r", encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print("loadJSON -> Ошибка: Файл не найден.")
    except json.JSONDecodeError:
        print("loadJSON -> Ошибка: Некорректный JSON.")
    except Exception as e:
        print(f"loadJSON -> Произошла непредвиденная ошибка: {e}")

# Функция обработки данных из JSON-файла
def processScenarioData(scenario_data, 
                        check_for_prepared=True, 
                        need_objects_preparation=False, 
                        save_prepared=False,
                        add_curtains=True,
                        scene_numbers=None, 
                        test_duration=None):
    try:
        scenario_title = scenario_data.get("scenario_title", "Untitled")
        scenario_title_for_path = scenario_data.get("scenario_title_for_path", "Untitled")
        # Создаем директорию для кадров
        scenario_path = f"performances/{scenario_title_for_path}"
        if not os.path.exists(scenario_path):
            os.makedirs(scenario_path)

        default_movie_sizes = [1280, 720]
        movie_sizes = scenario_data.get("movie_sizes", default_movie_sizes)

        default_main_font = {
            "font_name": "Nunito/static/Nunito-Medium.ttf",
            "font_size": 30,
            "font_color": "#ffffff",
            "stroke_color": "#000000", 
            "stroke_width": 1
        }
        default_accent_font = {
            "font_name": "Nunito/static/Nunito-Medium.ttf",
            "font_size": 30,
            "font_color": "#ebc354",
            "stroke_color": "#1a100b", 
            "stroke_width": 1
        }
        main_font = scenario_data.get("main_font", default_main_font)
        accent_font = scenario_data.get("accent_font", default_accent_font)

        scenes = scenario_data.get("scenes")
        scene_clips = []
        for scene in scenes: 

            index = scene.get("index")
            if scene_numbers:
                if index in scene_numbers:
                        duration = scene.get("duration") if not test_duration else test_duration
                else:
                    continue
            else:
                duration = scene.get("duration")

            # Фон
            background_image_name = scene.get("background")
            background_image_path = f"sources/images/backgrounds/{background_image_name}"

            background_clip = ImageClip(background_image_path).with_duration(duration).with_layer_index(0)
            background_image_sizes = background_clip.size

            clips = [background_clip]

            # Обработка объектов 
            composition = scene.get("composition", [])
            for i, element in enumerate(composition):

                category = element.get("category")
                folder = f"sources/images/{category}/"
                image_name = element.get("image_name")
                name_without_extension = os.path.splitext(os.path.basename(image_name))[0]
                extension = os.path.splitext(image_name)[1]
                object_image_path = folder + image_name

                # Подготовка вставляемого изображения
                object_image_loaded = False
                prepared_object_image_path = folder + name_without_extension + "_prepared" + extension
                if check_for_prepared:
                    if os.path.isfile(prepared_object_image_path):
                        prepared_object_image = Image.open(prepared_object_image_path)
                        object_image_loaded = True
                if not object_image_loaded:
                    if need_objects_preparation:
                        prepared_object_image = prepareObjectImage(object_image_path,
                                        threshold_flood=40,
                                        need_castom_Sobel=False,
                                        need_save=save_prepared,
                                        output_path=prepared_object_image_path)
                    else:
                        prepared_object_image = Image.open(object_image_path).convert("RGBA")
                
                if not prepared_object_image: 
                    raise Exception(f"Не удалось открыть файл объекта {image_name}.")

                # масштабирование
                scale = element.get("scale", 1.0)
                if scale <= 0: 
                    print(f"processScenarioData -> Недопустимое значение масштаба: {scale}")
                elif scale != 1.0:
                    prepared_object_image = prepared_object_image.resize((int(prepared_object_image.size[0]*scale), 
                                                                          int(prepared_object_image.size[1]*scale)),
                                                                          Image.Resampling.BICUBIC)
                
                # поворот
                rotation_angle = element.get("rotation_angle", 0)
                if rotation_angle != 0:
                    prepared_object_image = prepared_object_image.rotate(rotation_angle, resample=Image.Resampling.BICUBIC, expand=True) 
                
                # обрезка пустых краев
                prepared_object_image = prepared_object_image.crop(prepared_object_image.getbbox())

                # зеркальное отображение
                prepared_object_array = np.asarray(prepared_object_image)
                reflection = element.get("reflection", False)
                if reflection: 
                    prepared_object_array = np.flip(prepared_object_array, axis=1)
                
                # создаем из изображения оюъекта ImageClip
                image_clip = ImageClip(prepared_object_array).with_duration(duration) 
                
                # непрозрачность
                opacity = element.get("opacity", 1)
                if opacity < 0 or opacity > 1: 
                    print(f"processScenarioData -> Недопустимое значение непрозрачности: {opacity}")
                elif opacity != 1:
                    image_clip = image_clip.with_opacity(opacity)

                # ЭФФЕКТЫ
                for effect in element.get("effect", []):
                    if effect and (effect.get("effect_type", "none") != "none"):
                        image_clip = addEffect(image_clip, effect, duration)

                # АНИМАЦИЯ и позиционирование
                coordinates = ifRelativeToAbsoluteCoordinates(element.get("coordinates", [0, 0]), background_image_sizes)
                motion = element.get("motion", None)
                if motion and (motion.get("motion_type", "none") != "none"):
                    image_clip = animateElement(image_clip, motion, duration, coordinates, background_image_sizes)
                else: # Если нет движения, просто позиционируем элемент
                    image_clip = image_clip.with_position(coordinates)

                clips.append(image_clip.with_layer_index(element.get("z-index", i)))

            #Создание финального клипа сцены
            scene_clip = CompositeVideoClip(clips).resized(width=movie_sizes[0]).with_duration(duration)

            # ЭФФЕКТЫ СЦЕНЫ
            for scene_effect in scene.get("scene_effect", []):
                if scene_effect and (scene_effect.get("effect_type", "none") != "none"):
                    scene_clip = addEffect(scene_clip, scene_effect, duration)

            # Реплики
            lines = scene.get("lines", [])

            subtitle_clips = []
            last_disappear = 0.0
            for replica in lines:
                speaker = replica.get("character", "")
                text = replica.get("text", "")

                author_names = ["автор", "рассказчик", "author", "speaker", "storyteller", "narrator", ""]
                if speaker.lower() in author_names: 
                    source_font = main_font.copy()
                    default_source_font = default_main_font.copy()
                else:
                    source_font = accent_font.copy()
                    default_source_font = default_accent_font.copy()
                    text = f"({speaker}:) {text}"

                font = replica.get("font", source_font)
                font_name = font.get("font_name", source_font.get("font_name", default_source_font["font_name"]))
                font_path = f"sources/fonts/{font_name}"
                font_size = font.get("font_size", source_font.get("font_size", default_source_font["font_size"]))
                font_color = font.get("font_color", source_font.get("font_color", default_source_font["font_color"]))
                font_stroke_color = font.get("stroke_color", source_font.get("stroke_color", default_source_font["stroke_color"]))
                font_stroke_width = font.get("stroke_width", source_font.get("stroke_width", default_source_font["stroke_width"]))
                            
                appearing_time = replica.get("appearing_time", last_disappear)
                show_time = replica.get("time", duration - appearing_time)
                if show_time > duration - appearing_time:
                    show_time = duration - appearing_time
                
                last_disappear = appearing_time + show_time
                    
                # субтитры
                # Разбиваем длинный текст по строкам
                wrapped_text = wrapText(text, font_path, font_size, scene_clip.size[0]*0.9)

                replica_clip = (TextClip(font=font_path,
                                    # text=full_text, 
                                    text=wrapped_text,
                                    font_size=font_size,
                                    size=(movie_sizes[0], None),
                                    margin=(scene_clip.size[0]*0.2, scene_clip.size[1]*0.05),
                                    color=font_color,
                                    stroke_color=font_stroke_color, 
                                    stroke_width=font_stroke_width,
                                    method="caption",
                                    text_align="center")
                            .with_position(("center", "bottom"))
                            .with_start(appearing_time)
                            .with_duration(show_time))

                subtitle_clips.append(replica_clip) 

            clip = CompositeVideoClip([scene_clip, *subtitle_clips]).with_duration(duration)
            # # Если переключение между сценами
            # clip = FadeIn(0.5).apply(clip)
            # clip = FadeOut(0.5).apply(clip)
            scene_clips.append(clip)

        if not scene_clips:
            print(f"processScenarioData -> В итоговом видео нет ни одной сцены!")
            return
        final_movie = concatenate_videoclips(scene_clips, method="compose")
        
        if add_curtains:
            op_and_ed_duration = 5
            curtains_moving_time = 3.0
            raw_op = ImageClip(final_movie.get_frame(0)).with_duration(op_and_ed_duration).with_layer_index(0)
            raw_ed = ImageClip(final_movie.get_frame(final_movie.duration-0.01)).with_duration(op_and_ed_duration).with_layer_index(0)
            curtains_clip = ImageClip("sources/environ/curtains.png").resized(width=movie_sizes[0]).with_duration(op_and_ed_duration).with_layer_index(1)
            curtains_motion = [
                { 
                    "motion_type": "linear",
                    "starting_time": op_and_ed_duration - curtains_moving_time,
                    "params": {
                        "speed": [0, -curtains_clip.h/curtains_moving_time]
                    }
                },
                { 
                    "motion_type": "linear",
                    "starting_time": 0.0,
                    "motion_time": curtains_moving_time,
                    "params": {
                        "speed": [0, curtains_clip.h/curtains_moving_time]
                    }
                }
            ]
            curtains_clip_up = animateElement(curtains_clip, curtains_motion[0], op_and_ed_duration, [0, 0], final_movie.size)
            curtains_clip_down = animateElement(curtains_clip, curtains_motion[1], op_and_ed_duration, [0, -curtains_clip.h], final_movie.size)
            
            font_name = main_font.get("font_name", default_main_font["font_name"])
            text = [scenario_title, "Конец"]    
            op_ed_replica_clips = []  
            for i in range(2):  
                op_ed_replica_clips.append(TextClip(font=f"sources/fonts/{font_name}",
                                                    text=text[i],
                                                    font_size=80, # сделать регулируемым
                                                    size=(movie_sizes[0], None),
                                                    margin=(scene_clip.size[0]*0.2, scene_clip.size[1]*0.05),
                                                    color="#f2c572",
                                                    stroke_color="#753412", 
                                                    stroke_width=3,
                                                    method="caption",
                                                    text_align="center")
                                                .with_position(("center", "center"))
                                                .with_start(0.0)
                                                .with_duration(op_and_ed_duration)
                                                .with_layer_index(2))
            op_ed_replica_clips[0] = op_ed_replica_clips[0].with_effects([CrossFadeOut(curtains_moving_time)])
            op_ed_replica_clips[1] = op_ed_replica_clips[1].with_effects([CrossFadeIn(curtains_moving_time)])
            
            
            op = CompositeVideoClip([raw_op, curtains_clip_up, op_ed_replica_clips[0]]).with_duration(op_and_ed_duration)
            ed = CompositeVideoClip([raw_ed, curtains_clip_down, op_ed_replica_clips[1]]).with_duration(op_and_ed_duration)
            
            final_movie = concatenate_videoclips([op, final_movie, ed], method="compose")

        output_movie_path = f"{scenario_path}/{scenario_title_for_path}.mp4"
        final_movie.write_videofile(output_movie_path, fps=24)
        print(f"processScenarioData -> Итоговое видео сохранено: {output_movie_path}")
          
    except FileNotFoundError:
        print("processScenarioData -> Ошибка: Файл не найден.")
    # except json.JSONDecodeError:
    #     print("processScenarioData -> Ошибка: Некорректный JSON.")
    # except KeyError:
    #     print("processScenarioData -> Ошибка: Ключ не найден.")
    # except TypeError:
    #     print("processScenarioData -> Ошибка: Неверный тип данных.")
    # except Exception as e:
    #     print(f"processScenarioData -> Произошла ошибка: {e}")


def ifRelativeToAbsoluteCoordinates(coordinates, background_image_sizes):
    coordinates = np.array(coordinates)
    if np.all((0 <= np.abs(coordinates)) & (np.abs(coordinates) <= 1)):
        coordinates = [coordinates[0]*background_image_sizes[0], coordinates[1]*background_image_sizes[1]]
    return coordinates


def animateElement(clip, motion_data, scene_duration, coordinates, background_image_sizes):
    """
    Анимирует клип в соответствии с заданными параметрами движения.

    Args:
        clip (ImageClip):  Клип для анимации.
        motion_data (dict):  Параметры движения из JSON.
        scene_duration (float): Длительность всей сцены.

    Returns:
        VideoClip: Анимированный видеоклип.
    """
    starting_time = motion_data.get("starting_time", 0.0)
    if starting_time >= scene_duration:
        return clip.with_position(clip.pos)
    motion_time = motion_data.get("motion_time", scene_duration-starting_time)  
    
    # Убеждаемся, что motion_time не превышает длительность сцены
    if motion_time > scene_duration - starting_time:
        motion_time = scene_duration - starting_time  
    
    start_x, start_y = ifRelativeToAbsoluteCoordinates(motion_data["params"].get("start", coordinates), 
                                                       background_image_sizes)
    
    ax, ay = motion_data["params"].get("acceleration", [0, 0])
    vx, vy = motion_data["params"].get("speed", [0, 0])
    omega = motion_data["params"].get("angle_speed", 0) 

    stop = motion_data["params"].get("stop", None)
    if stop: 
        stop_x, stop_y = ifRelativeToAbsoluteCoordinates(stop, background_image_sizes)
        vx = (stop_x - start_x) / motion_time
        vy = (stop_y - start_y) / motion_time
        ax, ay = 0, 0
    
    w, h = clip.size
    def angle(t):
        if starting_time <= t <= starting_time + motion_time:
            return omega * (t - starting_time)  # Угол поворота
        else:
            return 0 if t < starting_time else omega * motion_time # Максимальный угол
        
    def position(t):
        x, y = start_x, start_y
        if vx or vy or ax or ay:
            if starting_time <= t <= starting_time + motion_time:
                local_t = t - starting_time 
                x = start_x + vx * local_t + ax * local_t**2 / 2
                y = start_y + vy * local_t + ay * local_t**2 / 2
            else:
                # Стоим на месте до или после анимации
                if t < starting_time:
                    x = start_x
                    y = start_y
                else:
                    x = start_x + vx * motion_time + ax * motion_time**2 / 2
                    y = start_y + vy * motion_time + ay * motion_time**2 / 2
    
        dx = dy = 0
        if omega:
            angle_in_radians = np.deg2rad(angle(t))
            dx = w/2 * (np.abs(np.cos(angle_in_radians)) - 1) + h/2 * np.abs(np.sin(angle_in_radians))
            dy = w/2 * np.abs(np.sin(angle_in_radians)) + h/2 * (np.abs(np.cos(angle_in_radians)) - 1)
        return (-dx+x, -dy+y)

    if omega:
        return Rotate(angle, expand=True, center=(w/2, h/2)).apply(clip).with_position(position)
    elif vx or vy or ax or ay:
        return clip.with_position(position)
    else:
        print(f"animateElement -> Параметры для движения не указаны")
        return clip.with_position(clip.pos)  # Без движения, возвращаем как есть
    

def addEffect(clip, effect_data, scene_duration):
    effect_type = effect_data["effect_type"]
    effect_appearance_time = effect_data.get("effect_appearance_time", 0) 
    if effect_appearance_time >= scene_duration:
        return clip
    effect_duration = effect_data.get("effect_duration", 0.5) 
    till_the_end = False
    if effect_appearance_time + effect_duration >= scene_duration:
        effect_duration = scene_duration - effect_appearance_time
        till_the_end = True
    effect_end_time = effect_appearance_time + effect_duration

    if effect_type == "disappearance":
        if till_the_end:
            clip = clip.with_effects_on_subclip([CrossFadeOut(effect_duration)], effect_appearance_time)
        else:
            clip = clip.with_effects_on_subclip([CrossFadeOut(effect_duration)], effect_appearance_time, effect_end_time)     
            clip = clip.subclipped(0, effect_end_time) # обрезаем то что после      
            empty_clip = ImageClip(np.zeros((clip.h, clip.w, 4))).with_duration(scene_duration-effect_end_time)
            clip = concatenate_videoclips([clip, empty_clip], method="compose")
        return clip
    elif effect_type == "appearance":
        if till_the_end:
            clip = clip.with_effects_on_subclip([CrossFadeIn(effect_duration)], effect_appearance_time)
        else:
            clip = clip.with_effects_on_subclip([CrossFadeIn(effect_duration)], effect_appearance_time, effect_end_time)
        clip = clip.subclipped(effect_appearance_time) # обрезаем то что до появления
        empty_clip = ImageClip(np.zeros((clip.h, clip.w, 4))).with_duration(effect_appearance_time)
        clip = concatenate_videoclips([empty_clip, clip], method="compose")
        return clip
    elif effect_type in ["scene_fade_in", "scene_fade_out", "scene_flash"]: # эффект для собранного клипа сцены: уход в выбранный цвет
        fade_color = effect_data.get("fade_color", [0, 0, 0])
        if effect_type == "scene_fade_in":
            if effect_data.get("in_the_op", False):
                clip = clip.with_effects([FadeIn(effect_duration, fade_color)])
            else:
                clip = clip.with_effects_on_subclip([FadeIn(effect_duration, fade_color)], 
                                                    effect_appearance_time, effect_appearance_time+effect_duration)
                clip = clip.subclipped(effect_appearance_time)
                color_array = np.array([[fade_color for _ in range(clip.w)] for _ in range(clip.h)])
                color_clip = ImageClip(color_array).with_duration(effect_appearance_time)
                clip = concatenate_videoclips([color_clip, clip], method="compose")
        else:
            if till_the_end or effect_data.get("in_the_end", False):
                clip = clip.with_effects([FadeOut(effect_duration, fade_color)])
            elif effect_type == "scene_flash":
                clip = clip.with_effects_on_subclip([FadeOut(effect_duration/4, fade_color)], 
                                                    effect_appearance_time, effect_appearance_time+effect_duration/4)
                clip = clip.with_effects_on_subclip([FadeIn(effect_duration/4, fade_color)], 
                                                    effect_appearance_time+effect_duration*3/4, effect_end_time)
                left_clip = clip.subclipped(0, effect_appearance_time+effect_duration/4)
                right_clip = clip.subclipped(effect_appearance_time+effect_duration*3/4)
                color_clip = ImageClip(np.array([[fade_color for _ in range(clip.w)] for _ in range(clip.h)])).with_duration(effect_duration/2)
                clip = concatenate_videoclips([left_clip, color_clip, right_clip], method="compose")
            else:
                clip = clip.with_effects_on_subclip([FadeOut(effect_duration, fade_color)], 
                                                    effect_appearance_time, effect_appearance_time+effect_duration)
                clip = clip.subclipped(0, effect_appearance_time+effect_duration)
                color_array = np.array([[fade_color for _ in range(clip.w)] for _ in range(clip.h)])
                color_clip = ImageClip(color_array).with_duration(scene_duration-(effect_appearance_time+effect_duration))
                clip = concatenate_videoclips([clip, color_clip], method="compose")
        return clip
    else:
        print(f"animateElement -> Неизвестный тип эффекта: {effect_type}")
        return clip # Возвращаем как есть 
    
def wrapText(text, font_path, font_size, max_width):
    # Разбивает текст по строкам, чтобы влез в заданную ширину
    font = ImageFont.truetype(font_path, font_size)
    lines = []
    words = text.split()
    line = ""
    for word in words:
        test_line = line + word + " "
        bbox = font.getbbox(test_line)
        w = bbox[2] - bbox[0]  # width = right - left
        if w <= max_width:
            line = test_line
        else:
            lines.append(line.strip())
            line = word + " "
    if line:
        lines.append(line.strip())
    return "\n".join(lines)