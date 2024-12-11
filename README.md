# Fire Vandalism Project

## Project Structure
Two dot files hold necessary values for the project to function
* .env
  * This file holds the x-api code needed to communicate with the backend server.

* .env.json
  * This file holds the unique uid of each camera and their associated RTSP links.

## Project building
This project is reliant on a mask being available so that the AI would have a way of knowing if the human alert should be triggered.  
This mask is obtained by taking a snapshot of the area and drawing a region of interest on the floor.  
The floor is mapped out (to preserve orientation) by selecting four corners in a clockwise (or anti-clockwise) manner and a point is selected to draw one or more Region of Interest. If the area needs to be larger the circumference of the circle can be enlarged.

Use the command: `python build_floor_mask.py --img_path <<camera scene>> --save_path <<csv file>> --size <<circle size (optional)>>`  
The csv file would then need to be compressed using command: `gzip --best -v <<csv file>>`

In the current project we are using the file `FloorMask.csv.gz` to hold the region of interest.  
Please note this would need to be remade if the camera position or orientation changes.

### Visualize the generated mask
Can be done using the command: `python visualize_floor_mask.py --img_path <<camera scene>> --mask_path <<csv file>>`

## Project Usage
The project can be activated using the commands:
`conda env create -f environment.yml`  

## Docker Container & Commands
Activate it using commands: 
* `docker exec -it fire_vandalism_detetction_gpu /bin/bash`.
* `cd home/fire_detection/`
* `tmux new-session -A -s servers_session`
* `conda activate yolov8`
* `python application.py`
### Debugging
* Due to the current implementation of python's threading library being faulty (see [stackoverflow](https://stackoverflow.com/questions/19155297/python-queues-memory-leaks-when-called-inside-thread)) we are resorting to restarting the app every three hours.  
`for i in {00..30000}; do timeout 180m python application.py; done`

* The model is hardcoded to rely on GPU (`cuda:1`). This may cause issues if the GPU crashes.  
A try catch loop may be implemented to test for GPU availablity, but it is best to restart docker and try again.


## Further Improvments
* Implement try catch system to test for GPU and act accordingly.
* Try solving the memory leakage and do away with the bash command.
* Improve model accuracy on Fire detetction.
  * It is my honest theory that we absolutely cannot improve the model unless we start training on actual cctv footage.  
  While this may cause issues in overconfidence during testing, the sheer lack of appropriate data will always remain a barrier.