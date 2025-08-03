# TO DO:
# рефакторинг разбить функции на вспомагательные
# объект настроек как параметр функций

from scenario_processing import loadJSON, processScenarioData

from moviepy import VideoFileClip
from moviepy.video.fx import MultiplySpeed

if __name__ == "__main__":
    scenario_file_name = input("Введите имя файла без расширения: ")
    # current options: happiness_in_a_box scenario_test" 
    scenario_file_path = f"scenarios/{scenario_file_name}.json"
    print(f"Вы выбрали файл {scenario_file_path}")
    scenario_data = loadJSON(scenario_file_path)
    if scenario_data: 
        add_curtains = True if input("Добавить театральный занавес? [y/n]: ") == "y" else False
        test_mode = True if input("Включить тестовый режим? [y/n]: ") == "y" else False

        needed_scenes = None
        test_duration = None
        if test_mode:
            needed_scenes = list(map(int, input("Введите номера сцен, которые необходимо зарендерить, через пробел: ").split()))
            test_duration = int(input("Введите длительность одной сцены (если необходимо зарендерить сцену полностью, нажмите 0): "))
        print("Here we go...")   
        processScenarioData(scenario_data,  
                            check_for_prepared=True,
                            need_objects_preparation=True, 
                            save_prepared=True, 
                            add_curtains=add_curtains,
                            scene_numbers=needed_scenes, 
                            test_duration=test_duration)
    print("done")