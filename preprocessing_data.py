import os
import shutil

base_path = "Dataset/"
session_index = 1

# Repeat from User1 to User21
for user_num in range(1, 22):
    user_folder = f"User{user_num}"
    user_path = os.path.join(base_path, user_folder)

    for session_num in range(1, 4):  
        session_folder = f"Session{session_num}"
        session_path = os.path.join(user_path, session_folder)

        if os.path.exists(session_path):
            new_session_path = os.path.join(base_path, f"Session{session_index}")
            shutil.move(session_path, new_session_path) 

            session_index += 1 

    # Removing User folder
    try:
        shutil.rmtree(user_path)
        print(f"Deleted User folder: {user_path}")
    except OSError as e:
        print(f"Error deleting {user_path}: {e}")
