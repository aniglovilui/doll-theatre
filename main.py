# TO DO:
# рефакторинг разбить функции на вспомагательные
# объект настроек как параметр функций

from scenario_processing import loadJSON, processScenarioData

from moviepy import VideoFileClip
from moviepy.video.fx import MultiplySpeed

if __name__ == "__main__":
    # filepath = "scenarios/happiness_in_a_box.json" 
    filepath = "scenarios/scenario_test.json"
    scenario_data = loadJSON(filepath)

    test_mode = True
    if test_mode:
        needed_scenes = [12]
        test_duration = 2
        processScenarioData(scenario_data,  
                        check_for_prepared=True,
                        need_objects_preparation=True, 
                        save_prepared=True, 
                        add_curtains=True,
                        scene_numbers=needed_scenes, 
                        test_duration=test_duration)
    else:
        print("Here we go...")
        processScenarioData(scenario_data,  
                        check_for_prepared=True,
                        need_objects_preparation=True, 
                        save_prepared=True,
                        add_curtains=True)

    # video_clip = VideoFileClip("performances/Счастье в шкатулке/Счастье в шкатулке.mp4")
    # # Ускоряем видео в 2 раза
    # video_clip_x2 = MultiplySpeed(factor=1.5).apply(video_clip)
    # # Сохраняем ускоренное видео
    # video_clip_x2.write_videofile("performances/Счастье в шкатулке/Счастье в шкатулке 1point5X.mp4", fps=video_clip.fps) # нбх указать fps оригинального видео
    print("done")