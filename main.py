from scenario_processing import loadJSON, processScenarioData


# need make paths with os.join
if __name__ == "__main__":
    scenario_file_name = input("Input scenario filename without extension: ")
    """
    current options: 
        happiness_in_a_box 
        scenario_test
    """
    scenario_file_path = f"scenarios/{scenario_file_name}.json"
    print(f"You chose file {scenario_file_path}")
    scenario_data = loadJSON(scenario_file_path)
    if scenario_data: 
        add_curtains = True if input("Add a theater curtain? [y/n]: ") == "y" else False
        test_mode = True if input("Enable test mode? [y/n]: ") == "y" else False

        needed_scenes = None
        test_duration = None
        if test_mode:
            needed_scenes = list(map(int, input("Enter the numbers of scenes you want to render, separated by spaces: ").split()))
            test_duration = int(input("Enter the duration of one scene (if you want to render the entire scene, enter 0): "))
        print("Processing...")   
        processScenarioData(
            scenario_data,
            check_for_prepared=True,
            need_objects_preparation=True, 
            save_prepared=True, 
            add_curtains=add_curtains,
            scene_numbers=needed_scenes, 
            test_duration=test_duration,
        )
    print("Done!")