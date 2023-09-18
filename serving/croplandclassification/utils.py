import time


def monitor_ee_tasks(task_list, check_interval=60):
    """
    Monitors a list of Earth Engine tasks until all tasks are completed or failed.

    Args:
        task_list (list): A list of Earth Engine tasks to monitor.
        check_interval (int): The interval in seconds between task status checks (default is 60 seconds).
    """
    # Check if the task list is empty
    if not task_list:
        print("No tasks to monitor.")
        return

    # Start the tasks
    for task in task_list:
        task.start()

    # Monitor the tasks
    while any(task.status()["state"] in ["READY", "RUNNING"] for task in task_list):
        print("Monitoring tasks...")
        time.sleep(check_interval)

    # Check the final status of each task
    for i, task in enumerate(task_list):
        if task.status()["state"] == "COMPLETED":
            print()
            print(f"Task {i + 1} completed successfully!\n")
            task_status = task.status()
            if "description" in task_status or "destination_uris" in task_status:
                print(f"File Description: {task_status['description']}\n")
                print(f"Destination url: {task_status['destination_uris']}\n")
        else:
            print(f"Task {i + 1} failed or was canceled.")