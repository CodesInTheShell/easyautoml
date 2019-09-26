import os
import config

def create_project_dir(proj_dir_name):

	dir_to_create = config.ROOT_DIR + '/models_trained/' + proj_dir_name

	if not os.path.exists(dir_to_create):
		os.mkdir(dir_to_create)
		print("Directory " , dir_to_create ,  " Created ")
		return dir_to_create
	else:    
		print("Directory " , dir_to_create ,  " already exists")
		return False